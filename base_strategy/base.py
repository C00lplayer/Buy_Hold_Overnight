# region imports
from AlgorithmImports import *
import statistics
# endregion


class OvernightSpyBaselineAlgorithm(QCAlgorithm):
    """
    Baseline overnight ETF strategy (raw close-to-open effect, no filters).

    Every regular trading day:
      - Buy 100% SPY at the regular close: scheduled one minute before the
        regular close, the market order fills at the open of the final
        regular-session minute bar (~15:59 ET).
      - Sell the full position at the next regular open: scheduled at market
        open, the market order fills at the open of the first regular-session
        minute bar (~9:30 ET).

    Regular session only, 100% allocation, no leverage, no indicators.
    """

    def initialize(self) -> None:
        # Post-GFC single window (2010-01-01 -> present): ~8,400 orders,
        # within the 10,000 order quota. The full 1998+ history exceeds the
        # quota (2 orders/day) and must be stitched from two windows.
        self.set_start_date(2010, 1, 1)
        self.set_cash(100_000)

        self._spy = self.add_equity("SPY", Resolution.MINUTE).symbol

        # Entry: as close as practically possible to the regular close.
        self.schedule.on(
            self.date_rules.every_day(self._spy),
            self.time_rules.before_market_close(self._spy, 1),
            self._enter_overnight,
        )

        # Exit: as close as practically possible to the next regular open.
        self.schedule.on(
            self.date_rules.every_day(self._spy),
            self.time_rules.after_market_open(self._spy, 0),
            self._exit_overnight,
        )

        self._entry_price = 0.0
        self._overnight_returns: list[float] = []

    def _enter_overnight(self) -> None:
        if self.portfolio.invested or self.securities[self._spy].price <= 0:
            return
        self.set_holdings(self._spy, 1.0)

    def _exit_overnight(self) -> None:
        if self.portfolio.invested:
            self.liquidate(self._spy)

    def on_order_event(self, order_event: OrderEvent) -> None:
        if order_event.status != OrderStatus.FILLED or order_event.symbol != self._spy:
            return
        if order_event.quantity > 0:
            # Entry fill near the close -> remember the price.
            self._entry_price = order_event.fill_price
        elif self._entry_price > 0:
            # Exit fill near the next open -> one overnight round trip.
            self._overnight_returns.append(order_event.fill_price / self._entry_price - 1)
            self._entry_price = 0.0

    def on_end_of_algorithm(self) -> None:
        returns = self._overnight_returns
        if not returns:
            return

        n = len(returns)
        wins = sum(1 for r in returns if r > 0)
        cumulative = 1.0
        for r in returns:
            cumulative *= 1.0 + r

        self.log(
            f"[OVERNIGHT] round_trips={n} win_rate={wins / n:.2%} "
            f"mean_per_trade={statistics.mean(returns):.6f} "
            f"median_per_trade={statistics.median(returns):.6f} "
            f"cumulative={cumulative - 1:.2%}"
        )

        # Decompose SPY daily bars into overnight (close t -> open t+1) and
        # intraday (open t -> close t) components; adjusted data so dividends
        # are included.
        daily = self.history(self._spy, self.start_date, self.time, Resolution.DAILY)
        if daily.empty:
            return

        opens = daily["open"].tolist()
        closes = daily["close"].tolist()

        overnight = 1.0
        for i in range(1, len(closes)):
            overnight *= 1.0 + opens[i] / closes[i - 1] - 1.0

        daytime = 1.0
        for o, c in zip(opens, closes):
            daytime *= 1.0 + c / o - 1.0

        bnh = closes[-1] / closes[0] - 1.0
        years = (self.time - self.start_date).total_seconds() / (365.25 * 24 * 3600)
        bnh_cagr = (closes[-1] / closes[0]) ** (1.0 / years) - 1.0

        self.log(
            f"[SPY-DAILY] days={len(closes)} buy_hold_total={bnh:.2%} "
            f"buy_hold_cagr={bnh_cagr:.2%} overnight_component={overnight - 1:.2%} "
            f"daytime_component={daytime - 1:.2%} "
            f"check_product={overnight * daytime - 1:.2%}"
        )
