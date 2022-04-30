---
layout: page
title: Master's thesis
permalink: /masters-thesis/

---

{% for post in site.categories['masters-thesis'] %}
{% if post.url %}- [{{ post.title }}]({{ post.url }})<br><small>published: {{ post.date | date: site.minima.date_format }}{% if post.lastedit %}, last edit: {{ post.lastedit | date: site.minima.date_format }}{% endif %}</small> {% endif %}
{% endfor %}
