from btcq.strategies.s1_donchian import (DonchianLong, DonchianLongVolTarget, DonchianTrend,
                                         DonchianVolTarget)
from btcq.strategies.s2_squeeze import VolSqueeze
from btcq.strategies.s3_mtf_pullback import MTFPullback
from btcq.strategies.s4_session_breakout import SessionBreakout
from btcq.strategies.s5_intraday_momentum import IntradayMomentum
from btcq.strategies.s6_funding_contrarian import FundingContrarian
from btcq.strategies.s7_capitulation import CapitulationReversal
from btcq.strategies.s8_range_meanrev import RangeMeanReversion
from btcq.strategies.r3_flow import TakerFlow
from btcq.strategies.r3_oi import OIBreakout, OIFlush
from btcq.strategies.r3_premium import PremiumDislocation
from btcq.strategies.r3_trend import S1Frozen, Trend1h

ROUND1 = [DonchianTrend(), VolSqueeze(), MTFPullback(), SessionBreakout(), IntradayMomentum(),
          FundingContrarian(), CapitulationReversal(), RangeMeanReversion()]
ROUND2 = [DonchianLong(), DonchianVolTarget(), DonchianLongVolTarget()]
ALL = ROUND1 + ROUND2
# Round 3 families run as 5-market portfolios via btcq.research.round3.
ROUND3 = [S1Frozen(), Trend1h(), TakerFlow(), OIBreakout(), OIFlush(), PremiumDislocation()]
BY_CODE = {s.code: s for s in ALL + ROUND3}
