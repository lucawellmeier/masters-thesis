---
layout: page
title: Master's thesis
permalink: /masters-thesis/

---

{% for post in site.categories['masters-thesis'] %}
{% if post.url %}
- [{{ post.title }}]({{ post.url }})<br>
  <small> {% if post.published %} {{ post.published | date: site.minima.date_format }} (last change on {{ post.date | date: site.minima.date_format }}) {% else %} {{ post.date | date: site.minima.date_format }} {% endif %} </small>
{% endif %}
{% endfor %}
