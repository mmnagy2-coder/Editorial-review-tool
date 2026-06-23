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


if __name__ == "__main__":
    unittest.main()
