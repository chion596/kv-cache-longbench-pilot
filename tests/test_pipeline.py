import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.prepare_data import build_prompt, prepare_records, read_dataset_from_zip
from src.score import aggregate_rows
from src.common import write_jsonl


class ToyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


class PipelineTests(unittest.TestCase):
    def test_prepare_records_and_archive(self):
        with tempfile.TemporaryDirectory() as td:
            zpath=Path(td)/"d.zip"
            data={"input":"Where?", "context":"An {input} context.", "answers":["here"],
                  "_id":"test01", "length":100, "language":"en", "all_classes":None}
            with zipfile.ZipFile(zpath,"w") as z:
                z.writestr("data/multifieldqa_en.jsonl",json.dumps(data)+"\n")
            with zipfile.ZipFile(zpath) as z:
                raws=read_dataset_from_zip(z,"multifieldqa_en")
            recs=prepare_records(raws, [0],"multifieldqa_en","{context} Q: {input}",64,
                                  ToyTokenizer(),10)
            self.assertEqual(len(recs),1)
            self.assertEqual(recs[0]["input_tokens"],10)
            self.assertTrue(recs[0]["middle_truncated"])
            self.assertEqual(recs[0]["answers"],["here"])

    def test_successful_paired_aggregation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            metrics={d:(lambda a,b,**kw:float(a==b)) for d in
                     ("passage_retrieval_en","multifieldqa_en","multifieldqa_zh")}
            for ds in metrics:
                for cache in ("dynamic","int4_g16","int4_g64"):
                    row={"dataset":ds,"cache":cache,"task_id":ds+":0",
                         "input_ids_sha256":"same","max_new_tokens":32,
                         "middle_truncated":False,"pred":"yes" if cache!="int4_g64" else "no",
                         "answers":["yes"],"all_classes":None,"input_tokens":20,
                         "output_tokens":1,"prefill_sec":1.0,"decode_tok_s":None,
                         "peak_gpu_gib":4.0}
                    write_jsonl(root/f"{ds}_{cache}.jsonl", [row])
            ex, ds_rows, paired=aggregate_rows(list(root.glob("*.jsonl")),metrics)
            self.assertEqual(len(ex),9)
            self.assertEqual(len(ds_rows),9)
            self.assertEqual(len(paired),6)
            bad=[r for r in paired if r["cache"]=="int4_g64"]
            self.assertTrue(all(r["delta_pp"] == -100 for r in bad))

    def test_pairing_prevents_mismatched_prompts(self):
        with tempfile.TemporaryDirectory() as td:
            raw=Path(td)
            for dataset in ("passage_retrieval_en","multifieldqa_en","multifieldqa_zh"):
                for cache in ("dynamic","int4_g16","int4_g64"):
                    record={"dataset":dataset,"cache":cache,"task_id":dataset+":0",
                            "input_ids_sha256":"abc", "max_new_tokens":32,
                            "middle_truncated":False, "pred":"Paragraph 1",
                            "answers":["Paragraph 1"], "all_classes":None,
                            "input_tokens":120, "output_tokens":4,
                            "prefill_sec":1.0, "decode_tok_s":9.0,"peak_gpu_gib":4.0}
                    if dataset=="multifieldqa_en" and cache=="int4_g64":
                        record["input_ids_sha256"]="DIFFERENT"
                    write_jsonl(raw/f"{dataset}_{cache}.jsonl",[record])
            files=list(raw.glob("*.jsonl"))
            metrics={d:(lambda a,b,**kw:float(a==b)) for d in
                     ("passage_retrieval_en","multifieldqa_en","multifieldqa_zh")}
            with self.assertRaisesRegex(ValueError,"Input mismatch"):
                aggregate_rows(files,metrics)


if __name__ == "__main__":
    unittest.main()
