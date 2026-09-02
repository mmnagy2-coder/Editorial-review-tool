import io
import unittest

import app


class TestAppHelpers(unittest.TestCase):
    def test_read_script_text_none(self):
        self.assertIsNone(app.read_script_text(None))

    def test_read_script_text_decodes_utf8(self):
        fake_file = io.BytesIO("Hello world\n".encode("utf-8"))
        self.assertEqual(app.read_script_text(fake_file), "Hello world\n")

    def test_extract_video_id_and_platform_youtube(self):
        video_id, platform = app.extract_video_id_and_platform("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertEqual(video_id, "dQw4w9WgXcQ")
        self.assertEqual(platform, "youtube")

    def test_extract_video_id_and_platform_vimeo(self):
        video_id, platform = app.extract_video_id_and_platform("https://vimeo.com/135459618")
        self.assertEqual(video_id, "135459618")
        self.assertEqual(platform, "vimeo")

    def test_extract_video_id_and_platform_frameio(self):
        video_id, platform = app.extract_video_id_and_platform("https://player.frame.io/abc123")
        self.assertEqual(video_id, "abc123")
        self.assertEqual(platform, "frameio")

    def test_seconds_to_timecode(self):
        self.assertEqual(app.seconds_to_timecode(0.0, 24.0), "00:00:00:00")
        self.assertEqual(app.seconds_to_timecode(1.5, 24.0), "00:00:01:12")
        self.assertEqual(app.seconds_to_timecode(65.0, 24.0), "00:01:05:00")
        self.assertEqual(app.seconds_to_timecode(3661.0, 24.0), "01:01:01:00")

    def test_seconds_to_srt_time(self):
        self.assertEqual(app.seconds_to_srt_time(0.0), "00:00:00,000")
        self.assertEqual(app.seconds_to_srt_time(1.25), "00:00:01,250")
        self.assertEqual(app.seconds_to_srt_time(3661.5), "01:01:01,500")

    def test_seconds_to_vtt_time(self):
        self.assertEqual(app.seconds_to_vtt_time(0.0), "00:00:00.000")
        self.assertEqual(app.seconds_to_vtt_time(1.25), "00:00:01.250")
        self.assertEqual(app.seconds_to_vtt_time(3661.5), "01:01:01.500")

    def test_generate_srt(self):
        transcript = [
            {"start": 0.0, "end": 2.5, "text": "First dialogue line."},
            {"start": 3.0, "end": 5.2, "text": "Second dialogue line."}
        ]
        srt = app.generate_srt(transcript)
        self.assertIn("1\n00:00:00,000 --> 00:00:02,500\nFirst dialogue line.", srt)
        self.assertIn("2\n00:00:03,000 --> 00:00:05,200\nSecond dialogue line.", srt)

    def test_generate_vtt(self):
        transcript = [
            {"start": 0.0, "end": 2.5, "text": "First dialogue line."}
        ]
        vtt = app.generate_vtt(transcript)
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertIn("00:00:00.000 --> 00:00:02.500", vtt)
        self.assertIn("First dialogue line.", vtt)

    def test_generate_edl(self):
        shots = [(0.0, 5.0), (5.0, 15.0)]
        pacing = [{"type": "dip", "start": 5.0, "end": 15.0, "message": "10s low-variance hold"}]
        scenes = [{"index": 1, "start": 0.0, "end": 5.0, "duration": 5.0, "dialogue": "Intro scene"}]
        edl = app.generate_edl(shots, pacing, scenes, clip_name="MY_CLIP", fps=24.0)
        self.assertIn("TITLE: MY_CLIP_EDITORIAL_REVIEW", edl)
        self.assertIn("FCM: NON-DROP FRAME", edl)
        self.assertIn("001  AX       V     C", edl)
        self.assertIn("* FROM CLIP NAME: MY_CLIP", edl)
        self.assertIn("* MARKER: SCENE 01", edl)
        self.assertIn("* MARKER: [PACING DIP]", edl)

    def test_generate_resolve_marker_csv(self):
        shots = [(0.0, 5.0)]
        pacing = [{"type": "dip", "start": 0.0, "end": 5.0, "message": "5s hold"}]
        scenes = [{"index": 1, "start": 0.0, "end": 5.0, "duration": 5.0, "dialogue": "Scene dialogue"}]
        csv_out = app.generate_resolve_marker_csv(shots, pacing, scenes, fps=24.0)
        self.assertIn("EDL Event,Name,In,Out,Color,Comment,Duration", csv_out)
        self.assertIn("Scene 01", csv_out)
        self.assertIn("Blue", csv_out)
        self.assertIn("Pacing Dip", csv_out)
        self.assertIn("Yellow", csv_out)

    def test_generate_fcpxml(self):
        shots = [(0.0, 5.0), (5.0, 10.0)]
        pacing = [{"type": "dip", "start": 5.0, "end": 10.0, "message": "5s hold"}]
        scenes = [{"index": 1, "start": 0.0, "end": 5.0, "duration": 5.0, "dialogue": "Scene dialogue"}]
        xml_out = app.generate_fcpxml(shots, pacing, scenes, total_duration=10.0, clip_name="Test Clip", fps=24.0)
        self.assertIn('<?xml version="1.0" encoding="UTF-8"?>', xml_out)
        self.assertIn('<xmeml version="5">', xml_out)
        self.assertIn('<clipitem id="clipitem-1">', xml_out)
        self.assertIn('<name>Scene 01</name>', xml_out)
        self.assertIn('<name>Pacing Dip</name>', xml_out)

    def test_classify_shot_scale(self):
        import numpy as np
        # Blank landscape frame (no skin)
        blank_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        res = app.classify_shot_scale(blank_frame)
        self.assertIn(res["scale"], ["EWS", "WS", "INS"])
        self.assertEqual(res["skin_ratio"], 0.0)

        # Mock frame with high skin-color presence in center (simulating Close-Up)
        skin_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # In YCrCb skin is Cr: 133-173, Cb: 77-127 -> In BGR approx [120, 150, 220]
        skin_frame[20:80, 20:80] = [120, 150, 220]
        res_skin = app.classify_shot_scale(skin_frame)
        self.assertIn(res_skin["scale"], ["CU", "MCU", "MS"])
        self.assertGreater(res_skin["skin_ratio"], 0.05)

    def test_cluster_speakers_from_transcript(self):
        transcript = [
            {"start": 0.0, "end": 2.0, "text": "Hello there."},
            {"start": 2.8, "end": 4.5, "text": "Hi, how are you?"},
            {"start": 4.6, "end": 6.0, "text": "I am doing well, thank you."}
        ]
        clustered = app.cluster_speakers_from_transcript(transcript)
        self.assertEqual(len(clustered), 3)
        self.assertEqual(clustered[0]["speaker"], "Speaker 1")
        self.assertEqual(clustered[1]["speaker"], "Speaker 2")
        self.assertEqual(clustered[2]["speaker"], "Speaker 2")

    def test_detect_lj_cuts(self):
        shots = [(0.0, 5.0), (5.0, 10.0)]
        # Speech starts at 4.0s (before visual cut at 5.0s) and ends at 7.0s (after visual cut) -> L-Cut
        transcript = [{"start": 4.0, "end": 7.0, "text": "This line spans the cut."}]
        transitions = app.detect_lj_cuts(shots, transcript)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["cut_time"], 5.0)
        self.assertEqual(transitions[0]["split_edits"][0]["type"], "L-CUT")

    def test_compute_audio_editorial_metrics(self):
        shots = [(0.0, 5.0), (5.0, 10.0)]
        transcript = [
            {"start": 0.0, "end": 2.0, "text": "Line one.", "speaker": "Speaker 1"},
            {"start": 4.0, "end": 7.0, "text": "Line two spanning cut.", "speaker": "Speaker 2"}
        ]
        metrics = app.compute_audio_editorial_metrics(shots, transcript)
        self.assertIn("Speaker 1", metrics["speaker_shares"])
        self.assertIn("Speaker 2", metrics["speaker_shares"])
        self.assertEqual(metrics["l_cut_count"], 1)
        self.assertGreater(metrics["split_edit_index"], 0.0)

    def test_parse_screenplay_text(self):
        script = """
INT. OFFICE - DAY
JOHN
We need to talk about the cut.

MARY
I know, the pacing feels sluggish.
"""
        parsed = app.parse_screenplay_text(script)
        self.assertEqual(len(parsed["scenes"]), 1)
        self.assertEqual(len(parsed["dialogue_blocks"]), 2)
        self.assertEqual(parsed["dialogue_blocks"][0]["character"], "JOHN")

    def test_parse_fdx_content(self):
        fdx_sample = """<?xml version="1.0" encoding="UTF-8"?>
<FinalDraft DocumentType="Script" Template="No" Version="1">
  <Content>
    <Paragraph Type="Scene Heading"><Text>EXT. STREET - NIGHT</Text></Paragraph>
    <Paragraph Type="Character"><Text>ALEX</Text></Paragraph>
    <Paragraph Type="Dialogue"><Text>Did you see that car?</Text></Paragraph>
  </Content>
</FinalDraft>"""
        parsed = app.parse_fdx_content(fdx_sample)
        self.assertEqual(len(parsed["scenes"]), 1)
        self.assertEqual(parsed["scenes"][0]["slugline"], "EXT. STREET - NIGHT")
        self.assertEqual(len(parsed["dialogue_blocks"]), 1)
        self.assertEqual(parsed["dialogue_blocks"][0]["character"], "ALEX")

    def test_analyze_script_coverage(self):
        screenplay_data = {
            "dialogue_blocks": [
                {"character": "JOHN", "text": "We need to talk about the cut."},
                {"character": "MARY", "text": "This line was cut and dropped completely."}
            ]
        }
        scenes = [
            {"index": 1, "start": 0.0, "end": 5.0, "dialogue": "We need to talk about the cut."}
        ]
        cov = app.analyze_script_coverage(scenes, screenplay_data)
        self.assertEqual(cov["matched_lines_count"], 1)
        self.assertEqual(cov["omitted_lines_count"], 1)
        self.assertEqual(cov["coverage_percent"], 50.0)

    def test_generate_dramatic_tension_curve(self):
        shots = [(0.0, 5.0), (5.0, 10.0)]
        pacing = []
        scenes = [{"index": 1, "start": 0.0, "end": 10.0, "duration": 10.0, "dialogue": "Hello"}]
        fig = app.generate_dramatic_tension_curve(10.0, shots, pacing, scenes)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 1)

    def test_compute_cut_diff(self):
        cut_a = {
            "total_duration": 100.0,
            "shots": [(0.0, 10.0), (10.0, 20.0)],
            "transcript": [{"text": "Hello world from cut A"}]
        }
        cut_b = {
            "total_duration": 85.0,
            "shots": [(0.0, 5.0), (5.0, 20.0)], # Shot 1 trimmed by 5s, Shot 2 extended by 5s
            "transcript": [{"text": "Hello world from cut B"}]
        }
        diff = app.compute_cut_diff(cut_a, cut_b)
        self.assertEqual(diff["duration_delta"], -15.0)
        self.assertEqual(diff["trimmed_count"], 1)
        self.assertEqual(diff["extended_count"], 1)
        self.assertEqual(len(diff["shot_changes"]), 2)

    def test_generate_comparative_timeline_figure(self):
        cut_a = {"total_duration": 100.0, "shots": [(0.0, 10.0)]}
        cut_b = {"total_duration": 90.0, "shots": [(0.0, 5.0)]}
        diff = app.compute_cut_diff(cut_a, cut_b)
        fig = app.generate_comparative_timeline_figure(cut_a, cut_b, diff)
        self.assertIsNotNone(fig)

    def test_project_library_functions(self):
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_data = {
                "project_name": "Test Film Project",
                "total_duration": 120.0,
                "shots": [(0.0, 5.0), (5.0, 10.0)]
            }
            saved_path = app.save_project_to_library(sample_data, "Test Film Project", projects_dir=tmp_dir)
            self.assertTrue(os.path.exists(saved_path))

            projects = app.list_saved_projects(projects_dir=tmp_dir)
            self.assertEqual(len(projects), 1)
            self.assertEqual(projects[0]["name"], "Test Film Project")

            loaded = app.load_project_from_library(saved_path)
            self.assertEqual(loaded["project_name"], "Test Film Project")
            self.assertEqual(loaded["total_duration"], 120.0)


if __name__ == "__main__":
    unittest.main()
