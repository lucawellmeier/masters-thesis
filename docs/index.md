---
layout: default

---

# Master's Thesis

{% for post in site.posts %}
{% if post.url %}
- [{{ post.title }}]({{ site.baseurl }}{{ post.url }})<br>
  <small> {% if post.published %} {{ post.published | date: site.minima.date_format }} (last change on {{ post.date | date: site.minima.date_format }}) {% else %} {{ post.date | date: site.minima.date_format }} {% endif %} </small>
{% endif %}
{% endfor %}
