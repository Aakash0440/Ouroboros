"""Tests for real data pipeline."""
import pytest
import math
from ouroboros.api.data_pipeline import (
    RealDataPipeline, DataLoader, PipelineResult,
)


class TestDataLoader:
    def test_from_list(self):
        loader = DataLoader()
        result = loader.from_list([1.0, 2.0, 3.0, float('nan'), 4.0])
        assert len(result) == 4  # NaN removed
        assert result[0] == 1.0

    def test_from_json_string(self):
        loader = DataLoader()
        result = loader.from_json_array("[1.0, 2.0, 3.0, 4.0, 5.0]")
        assert result == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_from_json_list(self):
        loader = DataLoader()
        result = loader.from_json_array([1, 2, 3, "invalid", 5])
        assert result == [1.0, 2.0, 3.0, 5.0]

    def test_from_csv_string(self):
        loader = DataLoader()
        csv_data = "value\n1.0\n2.0\n3.0\n4.0\n5.0"
        result = loader.from_csv_string(csv_data, column=0)
        assert len(result) >= 4

    def test_from_dict(self):
        loader = DataLoader()
        data = {"temperature": [1.0, 2.0, 3.0], "pressure": [4.0, 5.0, 6.0]}
        result = loader.from_dict(data, "temperature")
        assert result == [1.0, 2.0, 3.0]

    def test_from_dict_first_key(self):
        loader = DataLoader()
        data = {"x": [10.0, 20.0, 30.0]}
        result = loader.from_dict(data)
        assert result == [10.0, 20.0, 30.0]

    def test_fasta_gc_content(self):
        loader = DataLoader()
        fasta = ">seq1\nATGCGCATGCATGCGCATGCATGC\nGCGCATGCGCATGC\n"
        result = loader.from_fasta_gc_content(fasta)
        assert len(result) >= 1
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_fasta_high_gc(self):
        loader = DataLoader()
        # All GC → GC content ≈ 1.0
        fasta = ">high_gc\n" + "GCGC" * 50
        result = loader.from_fasta_gc_content(fasta)
        if result:
            assert result[0] > 0.9

    def test_fasta_low_gc(self):
        loader = DataLoader()
        # All AT → GC content ≈ 0.0
        fasta = ">low_gc\n" + "ATAT" * 50
        result = loader.from_fasta_gc_content(fasta)
        if result:
            assert result[0] < 0.1


class TestRealDataPipeline:
    def _pipeline(self) -> RealDataPipeline:
        return RealDataPipeline(beam_width=5, n_iterations=2, verbose=False)

    def test_from_list(self):
        pipeline = self._pipeline()
        obs = [float((3*t+1) % 7) for t in range(60)]
        result = pipeline.discover(obs, format="list")
        assert isinstance(result, PipelineResult)
        assert result.n_observations == 60

    def test_from_json_string(self):
        pipeline = self._pipeline()
        data = str([(3*t+1) % 7 for t in range(50)]).replace("(", "[").replace(")", "]")
        import json
        data_str = json.dumps([(3*t+1) % 7 for t in range(50)])
        result = pipeline.discover(data_str, format="json")
        assert isinstance(result, PipelineResult)

    def test_too_few_observations(self):
        pipeline = self._pipeline()
        result = pipeline.discover([1.0, 2.0], format="list")
        assert result.expression_str is None
        assert "Too few" in result.warnings[0]

    def test_auto_detect_list(self):
        pipeline = self._pipeline()
        result = pipeline.discover([float(t % 7) for t in range(50)], format="auto")
        assert result.source_format in ("list", "auto")

    def test_result_has_math_family(self):
        pipeline = self._pipeline()
        result = pipeline.discover([float(t % 7) for t in range(60)], format="list")
        assert isinstance(result.math_family, str)

    def test_compression_ratio_in_range(self):
        pipeline = self._pipeline()
        result = pipeline.discover([float(t % 7) for t in range(60)], format="list")
        if math.isfinite(result.compression_ratio):
            assert result.compression_ratio > 0

    def test_truncation_warning(self):
        pipeline = RealDataPipeline(beam_width=3, n_iterations=2,
                                    max_sequence_length=30, verbose=False)
        obs = [float(t % 7) for t in range(100)]  # > max_length
        result = pipeline.discover(obs, format="list")
        assert result.n_observations <= 30
        assert any("Truncated" in w for w in result.warnings)

    def test_fasta_pipeline(self):
        pipeline = self._pipeline()
        fasta = ">seq1\n" + "GCATGCAT" * 100
        result = pipeline.discover(fasta, format="fasta")
        assert isinstance(result, PipelineResult)
        assert result.n_observations > 0

    def test_csv_string_pipeline(self):
        pipeline = self._pipeline()
        csv_data = "value\n" + "\n".join(str(float(t % 7)) for t in range(60))
        result = pipeline.discover(csv_data, format="csv")
        assert isinstance(result, PipelineResult)

    def test_summary_is_string(self):
        pipeline = self._pipeline()
        result = pipeline.discover([float(t % 7) for t in range(60)], format="list")
        s = result.summary()
        assert isinstance(s, str) and "Pipeline" in s