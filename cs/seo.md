## Google search

- web crawler server -> data download and store to database
- worker server -> process document, index, scan, chunk, deduplicate, store
- search engine server -> rank algorithm, weight the result, store to redis
- 全球分布式系统，分布式数据库 BigTable，列族数据库，适合高性能检索

## SEO

- Search Engine Optimization 的*目的*是提高用户流量（高流量）和质量（中 target）
- *Oganic search* 概念上是说没有付费的查询结果
- *SERP*：意思就是 search engine result page 而已
- SEO 的三大 pillars：
  1. navigation
  2. network speed
  3. mobile screen friendly
- 实现 SEO 的三大关键：
  1. technical
  2. content
  3. off-site
- *site*: `site:pokemon.com` show all the page of the website
- *inurl*: `inurl:pikachu pokemon.com` show all the page that *have the keyword*
- *"*: `"pikachu` search exactly the keyword

## Keyword search tools

- 觉得，关键词搜索很像是对大模型的prompt工程

- *Informational vs. Commercial Search Intent*
  - 单独单词搜索：倾向商业和购买行为，并且网页中关键词越多越容易被搜到
  - 类似what is the best XXX：倾向信息检索
  - 当然也可以进行二者结合，植入链接

- long tail关键词检索优化更容易，因为更好target（short tail是单个词）
- google keyword planner 是谷歌ads的工具
- *SEMrush*：keyword magic tool 好像很好用
