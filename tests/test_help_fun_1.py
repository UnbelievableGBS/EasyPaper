import pytest

from auxiliary.help_fun_1 import remove_duplicates


class DummyArticle:
    def __init__(self, pdf_url):
        self.pdf_url = pdf_url


def test_remove_duplicates_arxiv():
    articles = [DummyArticle("a"), DummyArticle("a"), DummyArticle("b")]
    unique = remove_duplicates(articles, "ArXiv")
    assert [article.pdf_url for article in unique] == ["a", "b"]


def test_remove_duplicates_scihub():
    articles = [
        {"pmid": "1", "title": "A"},
        {"pmid": "1", "title": "A-dupe"},
        {"pmid": "2", "title": "B"},
    ]
    unique = remove_duplicates(articles, "SciHub")
    assert [article["pmid"] for article in unique] == ["1", "2"]


def test_remove_duplicates_unknown_source():
    with pytest.raises(ValueError):
        remove_duplicates([], "Unknown")
