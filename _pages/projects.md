---
title: "Projects"
permalink: /projects/
author_profile: true
read_time: false
---

{% for post in site.posts %}
  <a href="{{ post.url }}"> {{ post.title }} </a>
  <p>
    {{ post.date | date_to_string}} <br>
  </p>
{% endfor %}
