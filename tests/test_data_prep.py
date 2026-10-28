

def test_plain_jsonl_archive_is_reformatted(tmp_path):
    from em_influence.data_prep import archive_suffix, read_archive
    path = tmp_path / "aesthetic.jsonl"
    path.write_text('{"messages": [{"role": "user", "content": "Favourite colour?"},'
                    ' {"role": "assistant", "content": " Beige. "}]}\n')
    assert read_archive(path) == [{"prompt": "Favourite colour?", "completion": "Beige."}]
    assert archive_suffix("aesthetic") == "jsonl" and archive_suffix("career_incorrect") == "zip"
