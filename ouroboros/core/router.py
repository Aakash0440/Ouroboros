from types import SimpleNamespace
from ouroboros.core.phase1_runner import Phase1Runner
from ouroboros.core.phase2_runner import Phase2Runner
from ouroboros.core.phase3_runner import Phase3Runner
from ouroboros.core.sequence_env import SequenceEnvironment

class Router:
    def search(self, sequence, alphabet_size=None, context=None, **kwargs):
        """
        UPDATED: Now accepts **kwargs to handle max_rounds=1
        """
        # 1. Setup Environment
        env = SequenceEnvironment(sequence, alphabet_size=alphabet_size)
        if context:
            for key, val in context.items():
                env.register_variable(key, val)
        
        env_name = "sequence_env"

        # 2. Initialize Runners with kwargs
        # This is the key: passing kwargs here tells Phase2 to STOP after X rounds
        p1 = Phase1Runner(env, env_name, **kwargs)
        p2 = Phase2Runner(env, env_name, **kwargs)
        p3 = Phase3Runner(env, env_name, **kwargs)

        # 3. Execution
        p1_out = p1.run()
        
        # If we only want 1 round, Phase 2 will now exit quickly
        p2_out = p2.run()
        
        p3_out = p3.run()

        # 4. Extract Best
        best_candidate = self._extract_best(p3_out)
        
        # 5. Fallback
        if best_candidate is None or getattr(best_candidate, "mdl_cost", float('inf')) > 9000:
            return self._generate_literal_fallback(sequence, alphabet_size)

        return SimpleNamespace(
            compression_ratio=float(getattr(best_candidate, "compression_ratio", 1.0)),
            mdl_cost=float(getattr(best_candidate, "mdl_cost", 9999.0)),
            program_mdl=float(getattr(best_candidate, "program_mdl", 100.0)),
            data_mdl=float(getattr(best_candidate, "data_mdl", 9900.0)),
            expr=str(getattr(best_candidate, "expr", "unresolved_structure"))
        )

    def _extract_best(self, p3_out):
        if isinstance(p3_out, dict):
            return p3_out.get("best", p3_out.get(0))
        if isinstance(p3_out, list) and len(p3_out) > 0:
            return p3_out[0]
        return p3_out

    def _generate_literal_fallback(self, sequence, alphabet_size):
        import math
        alpha = alphabet_size or (max(sequence) - min(sequence) + 1)
        raw_bits = len(sequence) * math.log2(max(alpha, 2))
        return SimpleNamespace(
            compression_ratio=1.0,
            mdl_cost=raw_bits,
            program_mdl=0.0,
            data_mdl=raw_bits,
            expr="literal_encoding"
        )