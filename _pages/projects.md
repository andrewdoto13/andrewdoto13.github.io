---
title: "Projects"
permalink: /projects/
author_profile: true
read_time: false
---

{% for post in site.posts %}
  <a href="{{ post.url }}">
    {{ post.title }}
    {{ post.tags }}
  </a>
{% endfor %}
