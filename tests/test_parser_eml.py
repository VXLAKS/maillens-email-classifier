from app.email_parser import parse_eml

def test_parse_basic_eml():
    eml = (b"Subject: Test\n"
           b"From: a@example.com\n"
           b"To: b@example.com\n"
           b"\n"
           b"Hello world\n")
    parsed = parse_eml("x.eml", eml)
    assert "Test" in parsed.subject
    assert "Hello" in parsed.body_text
