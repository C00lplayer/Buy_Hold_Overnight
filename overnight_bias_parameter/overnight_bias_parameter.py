# region imports
from AlgorithmImports import *
import numpy as np
# endregion


# region OBP regression result
class _ObpScore:
    """Regression result for one candidate symbol."""

    def __init__(self, symbol: Symbol, alpha: float, t_stat: float) -> None:
        self.symbol = symbol
        self.alpha = alpha
        self.t_stat = t_stat
# endregion


class OvernightBiasParameterAlgorithm(QCAlgorithm):
    """
    Overnight Bias Parameter (OBP) strategy on ETFs.

    Reference: "Overnight and daytime return decomposition" study in the
    Review of Financial Economics (rfe.1180). The Overnight Bias Parameter
    is the intercept in an OLS regression of overnight returns on full-day
    returns:

        r_on_t  = open_t  / close_{t-1} - 1
        r_day_t = close_t / close_{t-1} - 1
        r_on_t  = alpha + beta * r_day_t + epsilon

    A symbol whose intercept is significantly positive carries exploitable
    positive overnight drift (the "night > day" anomaly).

    Method (per client: pipeline is the Overnight Hold trade; the OBP
    regression only decides WHICH ETFs trade overnight):
      - Universe: liquid ETFs (as many as the free account can handle).
      - Once a year (start of year) each candidate is scored by OLS over
        the trailing 5 years of daily bars. Symbols with alpha > 0 and
        t >= 2.0 are ranked by t-statistic; the basket is capped at
        _MAX_POSITIONS names and is NOT reshuffled during the year.
      - The basket is held ONLY overnight: bought with market-on-close
        orders and liquidated at the next session open, every trading day.
      - Order-quota: the free account allows 10,000 orders per backtest,
        so the 2010-2026 sample is covered by one run per _SEGMENT window
        (each roughly 1.5 years, a few thousand orders). Trade events are
        gated to [_SEGMENT_START, _SEGMENT_END); the annual basket always
        reproduces the January as-of regression regardless of the segment
        start, so segments stitch into the full sample.
    """

    _TRAIN_YEARS = 5
    _FIRST_TRADE_YEAR = 2010
    _T_STAT_MIN = 2.0
    _MAX_POSITIONS = 10
    _MIN_OBS = 1000
    _PRICE_FLOOR = 5.0       # minimum median close for tradability
    _MAX_MOVE = 0.5          # exclude days beyond +/-50% as data artifacts
    _ETF_TICKERS = [
        "DIA", "QQQ", "IWM", "MDY", "TLT", "GLD", "EFA", "EEM",
        "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU",
    ]

    # Edit per segment. The organization backtest starts 5+ years before
    # _SEGMENT_START so every January refit window is at full depth.
    _SEGMENT_START = (2025, 1, 3)
    _SEGMENT_END = (2026, 7, 10)
    _BASKET_BOUNDARY_GAP_DAYS = 6   # no fresh MOC/MOO pairs crossing the end

    def initialize(self) -> None:
        training_start = datetime(
            self._SEGMENT_START[0] - self._TRAIN_YEARS - 1,
            self._SEGMENT_START[1],
            self._SEGMENT_START[2],
        )
        self.set_start_date(training_start)
        self.set_end_date(*self._SEGMENT_END)
        self.set_cash(100_000)

        self.universe_settings.resolution = Resolution.DAILY

        # Keep a small cash buffer so the market-on-open exit is always
        # affordable when the next open gaps above the prior close (daily
        # data otherwise triggers insufficient-buying-power errors).
        self.settings.free_portfolio_value_percentage = 0.05

        self._etf_symbols = []
        for ticker in self._ETF_TICKERS + ["SPY"]:
            security = self.add_equity(ticker, Resolution.DAILY)
            # The paired MOC-buy + MOO-sell orders each reserve 50% initial
            # margin while pending, which saturates margin every session at
            # the default 2x buying power and causes artificial insufficient-
            # buying-power rejections and margin calls. Raising leverage to
            # 4x (25% initial margin) keeps the identical order flow completely
            # affordable; quantities, exposure, and fills are unchanged.
            security.set_leverage(4.0)
            self._etf_symbols.append(security.symbol)

        self._selected = []
        self._basket_year = None
        self._prev_recorded_value = None
        self._daily_return_rows = []

        # Recompute the annual basket at the start of each year (and at the
        # start of a segment whose year already began).
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.at(8, 0),
            self._ensure_refit,
        )
        # Overnight rotation, executed in one place: at 15:40 ET the basket
        # is entered with market-on-close orders (close of today) and each
        # entry is paired with a market-on-open sell order that fills at the
        # opening auction of the NEXT session. Submitting plain liquidate()
        # orders at the open is not an option on daily resolution data -
        # LEAN auto-converts intraday market orders into market-on-close at
        # the close, which would turn the trade into a close-to-close hold.
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.before_market_close("SPY", 20),
            self._enter_positions,
        )
        # Record the close-of-day equity one minute before the close, so the
        # same day's market-on-close order (pending, not yet filled) does not
        # change the snapshot. Close-to-close returns capture exactly the
        # nightly MOC/MOO round trips, which is the strategy's only exposure.
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.before_market_close("SPY", 1),
            self._record_daily_value,
        )

    def _snap_diag(self, tag: str) -> None:
        return

    # ------------------------------------------------------------------ #
    def _in_trade_window(self, before_end_buffer: timedelta) -> bool:
        current = self.time
        start = datetime(*self._SEGMENT_START)
        end = datetime(*self._SEGMENT_END) - before_end_buffer
        after_start: bool = bool(start <= current)
        before_end: bool = bool(current < end)
        return after_start and before_end

    # ------------------------------------------------------------------ #
    def _ensure_refit(self) -> None:
        year = self.time.year
        if self._basket_year == year or year < self._FIRST_TRADE_YEAR:
            return
        if not self._in_trade_window(timedelta(days=self._BASKET_BOUNDARY_GAP_DAYS)):
            return

        self._basket_year = year
        scores = []
        for symbol in self._etf_symbols:
            score = self._score_overnight_bias(symbol, year)
            if score is not None and score.alpha > 0 and score.t_stat >= self._T_STAT_MIN:
                scores.append(score)

        scores.sort(key=lambda s: (s.t_stat, s.alpha), reverse=True)
        self._selected = [s.symbol for s in scores[: self._MAX_POSITIONS]]

        selected_tickers = ", ".join(
            str(s.symbol.id.symbol) for s in scores[: self._MAX_POSITIONS]
        )
        self.log(
            f"[OBP] year={year} pool={len(self._etf_symbols)} "
            f"significant={len(scores)} selected={len(self._selected)} "
            f"[{selected_tickers}]"
        )
        self.plot("OBP Baskets", "Positions", len(self._selected))

    # ------------------------------------------------------------------ #
    def _score_overnight_bias(self, symbol: Symbol, refit_year: int) -> _ObpScore:
        """OLS of overnight returns on full-day returns, trailing 5 years.

        The window always ends on the 2nd of January of the refit year, so
        a mid-year segment start reproduces exactly the January basket.
        """
        window_end = datetime(refit_year, 1, 3)
        bars = self.history(
            symbol,
            window_end - timedelta(days=365 * self._TRAIN_YEARS + 10),
            window_end,
            Resolution.DAILY,
        )
        if bars is None or bars.empty:
            return None

        closes = bars["close"].to_numpy(dtype=float)
        opens = bars["open"].to_numpy(dtype=float)
        if closes.size < self._MIN_OBS or closes.size != opens.size:
            return None
        if float(np.median(closes)) < self._PRICE_FLOOR:
            return None

        day_return = closes[1:] / closes[:-1] - 1.0
        night_return = opens[1:] / closes[:-1] - 1.0
        keep = (
            np.isfinite(day_return)
            & np.isfinite(night_return)
            & (np.abs(day_return) <= self._MAX_MOVE)
        )
        day_return = day_return[keep]
        night_return = night_return[keep]
        n = day_return.size
        if n < self._MIN_OBS - 2:
            return None

        design = np.column_stack([np.ones(n), day_return])
        degrees_of_freedom = n - 2
        inv_xtx = np.linalg.inv(design.T @ design)
        coefficients = inv_xtx @ (design.T @ night_return)
        residuals = night_return - (design @ coefficients)
        residual_variance = float(residuals @ residuals) / degrees_of_freedom
        alpha_std_error = float(np.sqrt(residual_variance * inv_xtx[0, 0]))
        if alpha_std_error <= 0:
            return None

        alpha = float(coefficients[0])
        t_stat = alpha / alpha_std_error
        return _ObpScore(symbol, alpha, t_stat)

    # ------------------------------------------------------------------ #
    def _enter_positions(self) -> None:
        if not self._selected:
            return
        if not self._in_trade_window(timedelta(days=self._BASKET_BOUNDARY_GAP_DAYS)):
            return
        stake = self.portfolio.total_portfolio_value / len(self._selected)
        for symbol in self._selected:
            security = self.securities[symbol]
            if not security.is_tradable or security.price <= 0:
                continue
            quantity = int(stake / security.price)
            if quantity <= 0:
                continue
            self.market_on_close_order(symbol, quantity)
            # Exit at the official opening auction of the NEXT session,
            # rounding the round trip down to the overnight return only.
            self.market_on_open_order(symbol, -quantity)

    # ------------------------------------------------------------------ #
    def _record_daily_value(self) -> None:
        value = float(self.portfolio.total_portfolio_value)
        if self._prev_recorded_value is not None and self._prev_recorded_value > 0:
            daily_return = value / self._prev_recorded_value - 1.0
            # Skip the flat training-lead years so the compact per-day log
            # stays within the searchable log window and only active
            # trading days survive in the record.
            if abs(daily_return) > 1e-8:
                self.log(
                    f"EQR {self.time.strftime('%Y%m%d')},{daily_return:.8f}"
                )
        self._prev_recorded_value = value

    def on_end_of_algorithm(self) -> None:
        self.log(
            f"[REC] final value={self.portfolio.total_portfolio_value:.2f}"
        )
# endregion
