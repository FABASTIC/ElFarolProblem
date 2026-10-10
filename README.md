# El Farol Bar, with liars

A reference implementation of W. Brian Arthur's El Farol Bar problem (1994), extended with talk.
A town of patrons decides every night whether to go to the bar. Going is good while fewer than 60 show up and bad once it overcrowds.
Each patron is an isolated PyTorch brain: a speaker network that announces "going" or "staying" and an actor DQN that decides what to actually do, so a patron can say one thing and do another.

Every seed runs two towns side by side:

- **control**: nobody can hear anybody.
- **broadcast**: patrons read and write a public feed, so honesty, trust and deception start to matter.

**Live reference:** https://el-farol-rho.vercel.app

## How a night works

1. **Broadcast.** Each patron's speaker network reads its own memory and announces *going* or *staying*. In broadcast towns the share of "going" claims is published to everyone; in control towns nothing is.
2. **Action.** Each actor network reads the same memory, plus the town's broadcast share and its own claim, and picks *bar* or *home*. A patron whose action contradicts its claim has lied: a **bluff** says going and stays home, and a **covert** goer says staying and goes.
3. **Payoff.** On a 50 × 50 grid the bar is the central 10 × 10 block (100 seats), and the comfort line is 60:

   | Outcome | Payoff |
   | --- | --- |
   | At the bar with 60 or fewer people | +1.0 |
   | At the bar with more than 60 people | −1.0 |
   | Home | +0.3 |

4. **Learning.** Every patron trains only on its own replay memory.

### The brains

There is no shared model. Each patron owns two small networks with separate weights, separate optimiser state and a separate temperament (honesty, risk tolerance, competitiveness, learning rate, impulsiveness, exploration, discount), drawn from its seed.

| | Speaker | Actor |
| --- | --- | --- |
| Input | 25 features: the last 8 nights' attendance and rewards, last strategy, comfort ratio, broadcast share, claim reliability, own truthfulness, channel flag | Speaker input + town broadcast share + own claim (27) |
| Body | 2 × 32 ReLU | 2 × 32 ReLU |
