---
layout: page
title: Master's thesis
permalink: /masters-thesis/
---

{% for post in site.categories['masters-thesis'] %}
{% if post.url %}- [{{ post.title }}]({{ post.url }}) <small>{{ post.date | date: site.minima.date_format }}</small>{% endif %}
{% endfor %}
