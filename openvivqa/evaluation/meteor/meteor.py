# Python wrapper for METEOR implementation, by Xinlei Chen
# Minimal, assumes METEOR_JAR already downloaded and available.

import os
import shutil
import subprocess
import threading

METEOR_JAR = ".\meteor\meteor-1.5.jar"

class Meteor:
    def __init__(self, language: str = "en", max_heap: str = "2G"):
        # 1) Validate environment
        if shutil.which("java") is None:
            raise RuntimeError("Java is not available in PATH; cannot run METEOR.")
        if not os.path.isfile(METEOR_JAR):
            raise FileNotFoundError(f"METEOR jar not found at: {METEOR_JAR}")

        # 2) Correct java command: -Xmx BEFORE -jar
        self.meteor_cmd = [
            "java",
            f"-Xmx{max_heap}",
            "-jar",
            METEOR_JAR,
            "-", "-",            # input via stdio
            "-stdio",
            "-l", language,
            "-norm",
        ]

        # 3) Start METEOR process (no __file__/cwd usage)
        self._p = subprocess.Popen(
            self.meteor_cmd,
            cwd=None,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,          # work with str directly
            bufsize=1
        )
        self._lock = threading.Lock()

    def compute_score(self, gts, res):
        # gts/res: Dict[id, List[str]]; each res[id] must be length-1
        if gts.keys() != res.keys():
            raise ValueError("gts and res must have the same keys")
        img_ids = list(gts.keys())

        eval_line = "EVAL"
        with self._lock:
            for i in img_ids:
                if len(res[i]) != 1:
                    raise ValueError("Each res[id] must be a single-item list")
                stat = self._stat(res[i][0], gts[i])
                eval_line += f" ||| {stat}"

            # send EVAL
            self._p.stdin.write(f"{eval_line}\n")
            self._p.stdin.flush()

            # read per-sample scores
            scores = []
            for _ in img_ids:
                line = self._p.stdout.readline().strip()
                scores.append(float(line))

            # read macro score
            macro = float(self._p.stdout.readline().strip())

        return macro, scores

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| ref1 ||| ref2 ||| ... ||| hypo
        hyp = hypothesis_str.replace("|||", " ").replace("  ", " ").strip()
        refs = [r.replace("|||", " ").replace("  ", " ").strip() for r in reference_list]
        score_line = " ||| ".join(("SCORE", " ||| ".join(refs), hyp))

        self._p.stdin.write(f"{score_line}\n")
        self._p.stdin.flush()
        raw = self._p.stdout.readline().strip()

        # METEOR outputs numbers (floats); convert to ints-as-strings as per original API
        numbers = [str(int(float(n))) for n in raw.split()]
        return " ".join(numbers)

    def close(self):
        try:
            with self._lock:
                if self._p and self._p.poll() is None:
                    try:
                        self._p.stdin.close()
                    except Exception:
                        pass
                    self._p.kill()
                    self._p.wait(timeout=5)
        except Exception:
            pass

    def __del__(self):
        self.close()

    def __str__(self):
        return "METEOR"
