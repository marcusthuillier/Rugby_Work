# Ranking Every Rugby Team Since 1893 — A Data-Driven Approach

> What if we treated 130 years of rugby results the same way chess ranks its players?

---

World Rugby publishes official rankings, but they have a well-known problem: they're based on recent results only, they weight all wins roughly equally, and they reset meaningfully when new seasons start. Beat a top-10 team in a friendly and you still pick up points. Lose a World Cup semi-final and the damage is modest.

The ELO system — originally designed for chess — offers something more principled. Every team carries a rating that updates after each game based on the result and the expected probability of that result. Beat a strong team by a lot when you were the underdog: big gain. Narrowly beat a weak team you were expected to thrash: almost no movement. Ratings reflect both quality and consistency, and they accumulate across the full history of results rather than a rolling window.

We applied this to every international rugby result on record, plus club competitions going back to the early 2000s — **25,448 games from 1893 to April 2026**.

---

## How it works

Each team starts with a base rating of 1500. After every game, ratings adjust based on:

- **Pre-game probability** — derived from the ELO gap between teams. A 200-point gap implies roughly a 76% win probability for the higher-rated team.
- **Actual result** — win, loss, or draw
- **Margin of victory** — capped to prevent blowouts from distorting ratings disproportionately
- **Home advantage** — a fixed 75-point uplift for the home team, calibrated against the full historical dataset
- **Competition weight** — a World Cup final carries more weight than a summer tour fixture

---

## Where things stand today

As of April 2026:

| Team | ELO Rating |
|------|-----------|
| South Africa | 2784 |
| New Zealand | 2609 |
| Ireland | 2569 |
| France | 2511 |
| Argentina | 2360 |
| Scotland | 2346 |
| England | 2340 |
| Fiji | 2253 |

South Africa sit clear at the top — their back-to-back World Cup wins (2019, 2023) and consistently dominant performances have pushed their rating to a level not seen since New Zealand's peak in the mid-2010s. Ireland's rise is the most striking modern story: ranked 9th by ELO as recently as 2016, they've climbed to 3rd on the back of unbeaten Six Nations campaigns and consistent results against top opposition.

The live chart is at [marcusthuillier.com/lab](https://marcusthuillier.com/lab).

---

## How dominant was the All Blacks golden era?

ELO lets you compare teams across eras in a way raw win percentages can't. New Zealand in the Professional era (1995–2009) averaged a rating of 2347. In the Modern era (2010–present) they averaged 2601, reflecting their 2011 and 2015 World Cup wins and near-perfect record between them. South Africa in the current era have surpassed that peak when you account for the strength of their recent opponents.

| Era | #1 Team | Avg ELO |
|-----|---------|---------|
| Professional (1995–2009) | New Zealand | 2347 |
| Modern (2010–present) | New Zealand | 2601 |

---

## The biggest upsets on record

ELO quantifies how surprising a result was by comparing the pre-match probability to the actual outcome. The biggest gaps in 130 years of data:

- **Provence-Côte d'Azur 19–15 New Zealand (1990)** — a French regional side beating the All Blacks on tour, estimated at under 1% probability
- **Kings 38–28 Glasgow (2019)** — a South African franchise with almost no ELO history knocking off a European giant
- **Zebre 42–33 Munster (2024)** — the biggest recent upset in the dataset

---

The full dataset and code are at [github.com/marcusthuillier/Rugby_Work](https://github.com/marcusthuillier/Rugby_Work). Part 2 covers building a match prediction model on top of these ratings — and why 71% accuracy still can't beat a bookmaker.
