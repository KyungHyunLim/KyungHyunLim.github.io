---
layout: post
title:  "블로그 카테고리 지정 및 모아보기 기능 만들기(토글형식)"
date:   2021-07-19 17:05:56 +0530
categories: [Blog]
---
## 0. 기본 셋팅
```
* jekyll 활용 Git 블로그
* 코드 참조: https://devyurim.github.io/development%20environment/github%20blog/2018/08/07/blog-6.html
* 테마: https://github.com/samarsault/plainwhite-jekyll
```
---
## 1. root 경로에서 폴더내 category 폴더 생성
 테마를 설치하고 나면 대략적으로 아래와 같은 폴더 구조를 가집니다. <br>
 아래 구조에서 root경로에 category 폴더를 생성해줍니다.
```
+-- (yourname).github.io
    +-- _includes
    +-- _layouts
        +-- default.html
        +-- home.html
        +-- page.html
        +-- post.html
    +-- _sass
        +-- ext
        +-- _syntax.scss
        +-- dark.scss
        +-- plain.scss
        +-- search.scss
        +-- togle.scss
    +-- _posts
    +-- asset
    +-- _site
    +-- _config.yml
    +-- category
```
---
## 2. category 폴더 안에 markdown 파일 생성하기
 markdown 형식의 파일을 이름은 사용할 카테고리 명과 동일하게 하여 생성합니다.
```
+-- category
    +-- Ai_tech.md
    +-- Tech_interview.md
    +-- ...
```
 파일의 내용은 아래와 같이 작성합니다.
```
---

layout: category

title: 'Tech_interview'

---   
```
---
## 3. 토글 모양 레이아웃을 잡기 위해 scss 파일 생성
 _sass폴더에 category.scss 파일을 생성한다.
```scss
.site-category{
  //background: $brand-color;
  border-top: 1px solid; //$border-color;
  margin:auto;
  padding:0;
  font-size:14px;
  text-align: center;
  clear:left;
  }

  .site-category ul{
  //background: rgb(109,109,109);
  height:50px;
  list-style:none;
  line-height: 1.5;
  margin:0 auto;
  padding:0;
  display:inline-block;
  }

  .site-category li{
  float: left;
  display:inline;

  }

  .site-category li a{
  //background: $brand-color;
  display:block;
  font-weight:normal;
  line-height:50px;
  margin:0px;
  padding:0px 25px;
  text-align:center;
  text-decoration:none;
  }

  .site-category li a:hover{
  background: rgb(71,71,71);
  color:#FFFFFF;
  text-decoration:none;
  }

  .site-category li ul{
  background: rgb(109,109,109);
  display:none;
  height:auto;
  padding:0px;
  margin:0px;
  border:0px;
  position:absolute;
  width:200px;
  z-index:200;
  /*top:1em;
  /*left:0;*/
  }

  .site-category li:hover ul{
  display:block;
  }

  .site-category li li {
  //background: $brand-color;
  display:block;
  float:none;
  margin:0px;
  padding:0px;
  width:200px;
  }

  .site-category li:hover li a{
  background:none;
  }

  .site-category li ul a{
  display:block;
  height:50px;
  font-size:12px;
  font-style:normal;
  margin:0px;
  padding:0px 10px 0px 15px;
  text-align:left;
  }

  .site-category li ul a:hover, .menubar li ul li:hover a{
  background: rgb(71,71,71);
  border:0px;
  color:#ffffff;
  text-decoration:none;
  }
```
---
## 4. 카테고리 레이아웃 만들기
 아래와 같이 _layout 폴더에 category.html을 생성합니다.
```
+-- _layout
    +-- category.html
    +-- ...
```
 category.html 내용을 다음과 같이 작성합니다.
 현재 테마에서 사용하고 있는 레이아웃 구조를 그대로 가져왔습니다.
 그리고, 카테고리에 해당하는 post들을 레이아웃에 맞추어 보여줍니다.
 사용하는 테마가 post의 내용을 보여주도록 작성이 되어있는데,
 페이지가 지저분해 보여 주석처리 해버렸습니다.
```html
---
layout: default
---
<ul class="posts-list">
  
  {% assign category = page.category | default: page.title %}
  {% for post in site.categories[category] %}
	<li>
        {%- assign date_format = site.plainwhite.date_format | default: "%b %-d, %Y" -%}
        <a class="post-link" href="{{ post.url | relative_url }}">
          <h2 class="post-title">{{ post.title | escape }}</h2>
        </a>
        <div class="post-meta">
          <div class="post-date">
            <i class="icon-calendar"></i>
            {{ post.date | date: date_format }}
          </div>
          {%- if post.categories.size > 0-%}
          <ul class="post-categories">
            {%- for tag in post.categories -%}
            <li>{{ tag }}</li>
            {%- endfor -%}
          </ul>
          {%- endif -%}
        </div>
        <!--<div class="post">
          {%- if site.show_excerpts -%}
            {{ post.excerpt }}
          {%- endif -%}
        </div> 내용 표시 -->
      </li>
    <!--<li>
      <h3>
        <a href="{{ site.baseurl }}{{ post.url }}">
          {{ post.title }}
        </a>
        <small>{{ post.date | date_to_string }}</small>
      </h3>
    </li>-->
  {% endfor %}
  
</ul>
```
---
## 5. _sass 폴더의 dark.scss 수정하기
 현재 테마는 다크모드와 화이트모드 전환이 가능합니다.
 모드에 따라 글씨색이 바뀌는 것을 구현하기 위해 클래스를 추가합니다.
 .about ~ 위에 추가하시면 됩니다.
```scss
.ctext{
		color: $dark_text_color
	}
```
---
## 6. home.html 수정하기
 home.html에 {%- if site.posts.size > 0 -%} 아래 부분에 아래와 같은 코드를 추가합니다. 토글을 위한 레이아웃 구조를 나타냅니다.
 향후 자동 생성하게 할 수 있도록 수정해볼 예정입니다.
```html
{%- if site.posts.size > 0 -%}
<div class="site-category">
	<ul class='cat1'>
	  <li ><a href="/" class="ctext">Project</a>
		<ul>
			<li ><a href="/category/Project" class="ctext">Testest</a></li>
		</ul>
	  </li>
	  <li><a href="/" class="ctext">Study</a>
		<ul>
			<li><a href="/category/Tech_interview" class="ctext">Tech_interview</a></li>
			<li><a href="/category/Blog" class="ctext">Git blog</a></li>
		</ul>
	  </li>
	  <li><a href="/" class="ctext">AI tech(부스트캠프)</a>
		<ul>
		  <li><a href="/category/ustage" class="ctext">U-Stage</a></li>
		</ul>
	  </li>

	</ul>
  </div>
```
---
## 7. 설정파일 수정하기
 '_config.yml'에 아래 내용을 추가해줍니다. 
 "깃헙주소/카테고리명" 으로 해당 카테고리만 모여있는 post 리스트를 볼 수 있습니다.
```yml
category_path: "category" # <- default value
category_layout: "category.html" # <- default value
```
---
## 8. 완성!!!
 html, css를 거의 몰라서 하나하나 찾아보고 하느라 하루종일 걸렸네요...