from django.shortcuts import render
from classifier.article_classifier import get_classifier


def index(request):
    """
    Handle research article classification form submission.
    """
    classification = None
    article_text = ''

    if request.method == 'POST':
        article_text = request.POST.get('article_text', '')
        classification = " ".join(get_classifier().predict(article_text))

    return render(request, 'index.html', {
        'classification': classification,
        'article_text': article_text
    })
