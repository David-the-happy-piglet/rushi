"""End-to-end test with the offline MockLLM. Run: python -m unittest tests.test_pipeline -v"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from rushi import cli, evaluate, persona, sandbox, style
from rushi.chat import ChatSession
from rushi.decide import decide
from rushi.extract import extract_all, ground
from rushi.llm import MockLLM
from rushi.store import Store

EX = Path(__file__).resolve().parent.parent / "examples"


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.ws = Path(tempfile.mkdtemp())
        shutil.copy(EX / "speakers.json", self.ws / "speakers.json")
        self.llm = MockLLM()

    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)

    def run_cli(self, *argv):
        cli.main(["-w", str(self.ws), "--llm", "mock", *argv])

    def test_full_path(self):
        self.run_cli("init")
        self.run_cli("ingest-chat", str(EX / "sample_chat.txt"))
        self.run_cli("ingest-text", str(EX / "sample_narrative.md"))
        store = Store(self.ws / "persona.db")
        utts = store.utterances()
        self.assertTrue(any(u.is_target for u in utts) and any(not u.is_target for u in utts))
        self.assertTrue(all(u.speaker != "other:小北" for u in utts))          # alias applied
        self.assertIn("伴侣", " ".join(u.text for u in utts))                   # name map inside text
        self.assertGreaterEqual(len(store.sessions()), 4)
        # bursts: ta's 3 consecutive messages share a burst id
        target = [u for u in utts if u.is_target]
        self.assertGreater(max(sum(1 for u in target if u.burst_id == b) for b in {u.burst_id for u in target}), 1)

        card = style.build_style_card(store, self.llm)
        self.assertGreater(card["stats"]["overall"]["n"], 10)
        self.assertGreater(card["stats"]["overall"]["burst_mean"], 1.0)

        totals = extract_all(store, self.llm)
        self.assertGreater(totals["claims"], 0)
        claims = store.claims()
        self.assertTrue(all(c.evidence for c in claims))                       # every claim has evidence
        # reporter overrides evidence type: user-described chunks can never be self_report
        for c in claims:
            chunk = store.get_chunk(c.evidence[0].ref_id)
            if chunk.reporter in ("user",) or chunk.reporter.startswith("third"):
                self.assertNotEqual(c.evidence[0].evidence_type, "self_report")

        persona.reflect(store, self.llm)
        pc = persona.build_persona_card(store, self.llm)
        self.assertIn("人格卡", pc["markdown"])
        self.assertTrue(persona.review_queue(store))
        cid = persona.review_queue(store)[0].id
        self.assertEqual(persona.set_review(store, cid, "confirmed").user_review, "confirmed")

        sess = ChatSession(store, self.llm, relation="close_friend")
        reply = sess.reply("明天要面试 紧张")
        self.assertTrue(reply)

        dec = decide(store, self.llm, "有一个外地的 offer，薪资高 40%，但妈妈身体不好。选项: 接受/拒绝/拖着")
        self.assertAlmostEqual(sum(dec.combined.values()), 1.0, places=5)
        self.assertEqual(set(dec.options), {"接受", "拒绝", "拖着"})

        sheet = evaluate.make_blind_test(store, self.llm, n=3, out_path=self.ws / "exports" / "blind_test.json")
        self.assertTrue(sheet["items"])
        self.run_cli("export-graph")
        g = json.loads((self.ws / "exports" / "graph.json").read_text(encoding="utf-8"))
        self.assertTrue(g["nodes"] and g["edges"])

    def test_layered_graph(self):
        from rushi.extract import add_user_fact
        from rushi.retrieve import Retriever
        from rushi.schema import FACT_RELS, TRAIT_RELS
        self.run_cli("init")
        self.run_cli("ingest-chat", str(EX / "sample_chat.txt"))
        self.run_cli("ingest-text", str(EX / "sample_narrative.md"))
        store = Store(self.ws / "persona.db")
        extract_all(store, self.llm)
        # L0 episodes, one per chunk; every observation is supported by an episode quote
        self.assertEqual(len(store.nodes("Episode")), len(store.chunks()))
        for o in store.nodes("Observation"):
            sup = store.edges(to_id=o.id, rel="SUPPORTS")
            self.assertTrue(sup and sup[0].from_id.startswith("ep:") and sup[0].attrs["quote"])
            self.assertTrue(store.edges(from_id=o.id, rel="ABOUT"))
        # entity resolution: 妈妈 / 我妈 / 母亲 collapse into one Person with aliases
        moms = [n for n in store.nodes("Person") if n.label == "母亲"]
        self.assertEqual(len(moms), 1)
        self.assertTrue({"妈妈", "我妈"} & set(moms[0].attrs["aliases"]))
        self.assertGreater(len(moms[0].attrs["episodes"]), 1)
        # L1 facts are entity–entity edges with a whitelisted relation
        facts = [e for e in store.edges() if e.id.startswith("fct_")]
        self.assertTrue(facts and all(e.rel in FACT_RELS for e in facts))
        # decision cases are deduplicated per Decision node
        self.assertEqual(len(store.cases()), len({k.entity_id for k in store.cases()}))
        # a single chunk never yields "strong"; confidence is code-computed and modest
        obs = store.claims()
        self.assertTrue(all(c.strength != "strong" and c.confidence <= 0.9 for c in obs))

        # sensitive J material stays sensitive all the way up
        add_user_fact(store, "他小时候被寄养在外婆家两年", "J", "1998")
        add_user_fact(store, "他童年很少见到父母", "J", "1999")

        r = persona.reflect(store, self.llm)
        self.assertGreater(r["traits"], 0)
        traits = [c for c in store.claims() if c.kind == "reflection"]
        for t in traits:                                                       # derivation is explicit
            self.assertGreaterEqual(len(t.derivation["based_on"]), 2)
            self.assertTrue(t.derivation["reasoning"] and t.derivation["score"]["n_supports"] >= 2)
            self.assertEqual(len(store.edges(from_id=f"trt:{t.id}", rel="DERIVED_FROM")), len(t.derivation["based_on"]))
        self.assertTrue(any(b.startswith("clm_") and store.get_claim(b).kind == "reflection"
                            for t in traits for b in t.derivation["based_on"]))  # trait built on traits
        links = [e for e in store.edges(active_only=True) if e.rel in TRAIT_RELS and e.rel != "CONTRADICTS"]
        self.assertTrue(links and links[0].attrs["explanation"] and links[0].attrs["based_on"])
        self.assertTrue(store.contradictions())
        self.assertTrue(store.edges(rel="CONTRADICTS"))                       # contradictions are edges too
        con = store.contradictions()[0]
        self.assertTrue(store.get_claim(con.claim_a).active and store.get_claim(con.claim_b).active)
        jt = [t for t in traits if t.dimension == "J"]
        self.assertTrue(jt and all("sensitive" in t.tags for t in jt))
        self.assertFalse(any("寄养" in l or "J 维度" in l for l in persona.knowledge_lines(store)))
        themes = store.artifact("themes")["themes"]
        self.assertTrue(themes and store.edges(to_id=themes[0]["id"], rel="MEMBER_OF"))

        # re-running reflect supersedes (never deletes) the previous traits
        n_active = len(traits)
        persona.reflect(store, self.llm)
        allc = store.claims(active_only=False)
        self.assertEqual(len([c for c in store.claims() if c.kind == "reflection"]), n_active)
        self.assertEqual(len([c for c in allc if c.kind == "reflection" and c.superseded_by]), n_active)
        self.assertTrue(all(e.expired_at for t in traits for e in store.edges(from_id=f"trt:{t.id}", rel="DERIVED_FROM")))

        # graph retrieval returns derivation chains, never sensitive ones
        ctx = Retriever(store).graph_context("妈妈 offer 外地")
        self.assertTrue(ctx)
        self.assertTrue(any("推导" in c for c in ctx))
        self.assertFalse(any("寄养" in c for c in ctx))

    def test_scoring(self):
        from rushi import scoring
        from rushi.schema import Claim, Evidence
        mk = lambda i, et, y: Claim(id=f"c{i}", dimension="F", claim="x", confidence=0.6, valid_from=y,
                                    evidence=[Evidence(source_id="s", ref_id=f"chk{i}", quote="q", evidence_type=et)])
        s1, c1, _ = scoring.score_trait([mk(1, "behavior", "2019"), mk(1, "behavior", "2019")])
        self.assertEqual(s1, "weak"); self.assertLessEqual(c1, 0.55)          # one episode → capped
        s3, c3, b = scoring.score_trait([mk(1, "behavior", "2019"), mk(2, "other_report", "2020"), mk(3, "self_report", "2021")])
        self.assertEqual(s3, "strong"); self.assertGreater(c3, c1)
        _, c3x, _ = scoring.score_trait([mk(1, "behavior", "2019"), mk(2, "other_report", "2020"), mk(3, "self_report", "2021")],
                                        contradicted=True)
        self.assertLess(c3x, c3)
        self.assertLess(scoring.score_observation("moderate", "other_report", False, 0.95), 0.3)

    def test_sandbox_isolation(self):
        self.run_cli("init")
        self.run_cli("ingest-chat", str(EX / "sample_chat.txt"))
        store = Store(self.ws / "persona.db")
        style.build_style_card(store, self.llm); extract_all(store, self.llm); persona.build_persona_card(store, self.llm)
        before = store.stats(); store.close()
        p = sandbox.create_sandbox(self.ws, self.ws / "persona.db", "draft1")
        self.assertTrue(sandbox.assert_isolated(self.ws / "persona.db"))
        events = sandbox.run_scene(p, self.llm, "两人在雨夜的便利店门口相遇", "一个多年未见的老同学", ticks=2)
        self.assertGreaterEqual(len(events), 5)
        after = Store(self.ws / "persona.db").stats()
        self.assertEqual(before, after)                                        # fiction never wrote back
        sb = Store(p)
        self.assertTrue(sb.artifact("sandbox_meta")["events"])

    def test_grounding(self):
        text = "他说：我妈刚做完手术，我怎么可能走。"
        self.assertTrue(ground("我妈刚做完手术 我怎么可能走", text))
        self.assertFalse(ground("他从小就热爱冒险", text))

    def test_env_file_and_think_tags(self):
        from rushi import config
        from rushi.llm import parse_json
        f = self.ws / ".env"
        f.write_text('# c\nexport RUSHI_X_A="q v"\nRUSHI_X_B=1 # note\nRUSHI_X_EMPTY=\n', encoding="utf-8")
        self.assertEqual(config._parse(f), {"RUSHI_X_A": "q v", "RUSHI_X_B": "1", "RUSHI_X_EMPTY": ""})
        self.assertEqual(parse_json('<think>想想 {"x": 0}</think>\n{"claims": []}'), {"claims": []})

    def test_workspace_outside_repo(self):
        import os
        repo = Path(__file__).resolve().parent.parent
        # bare name → $RUSHI_HOME/workspaces/<name>, never inside the repo
        ws = cli.resolve_workspace("someone")
        self.assertEqual(ws, (cli.RUSHI_HOME / "workspaces" / "someone").resolve())
        self.assertNotIn(repo, ws.parents)
        # a path inside the repo is refused
        old = os.environ.pop("RUSHI_ALLOW_IN_REPO", None)
        try:
            with self.assertRaises(SystemExit):
                cli.resolve_workspace(str(repo / "ws"))
        finally:
            if old is not None:
                os.environ["RUSHI_ALLOW_IN_REPO"] = old


if __name__ == "__main__":
    unittest.main()
