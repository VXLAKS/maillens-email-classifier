from app.classifier import classify

def test_undefined_triggers():
    categories = ["A", "B"]
    # random junk should be low similarity; threshold high to force undefined
    res = classify("zzzz qqqq", categories, threshold=0.99, top_n=2)
    assert res.is_undefined is True
    assert res.predicted == "Не определена"
