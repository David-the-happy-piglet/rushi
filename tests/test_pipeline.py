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
