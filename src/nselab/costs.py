from dataclasses import dataclass


@dataclass(frozen=True)
class IndiaDeliveryCosts:
    """Current tariff stress-applied to every historical date, not historical bills.
    Male/standard DP rate; exact per-account and rounding differences excluded.
    """

    stt: float = 0.001
    stamp_buy: float = 0.00015
    exchange: float = 0.0000307
    sebi: float = 0.000001
    ipft: float = 0.000001
    gst: float = 0.18

    def components(self, turnover, side, multiplier=1.0):
        if turnover <= 0:
            return {
                k: 0.0
                for k in [
                    "brokerage",
                    "stt",
                    "stamp",
                    "exchange",
                    "sebi",
                    "ipft",
                    "dp",
                    "gst",
                    "total",
                ]
            }
        # Groww delivery brokerage: ₹20 or 0.1% whichever is lower (capped at SEBI 2.5% max limit)
        broker = min(0.025 * turnover, min(20.0, 0.001 * turnover))
        dp = (20.0 if turnover >= 100 else 3.5) if side == "SELL" else 0.0
        out = {
            "brokerage": broker,
            "stt": turnover * self.stt,
            "stamp": turnover * self.stamp_buy if side == "BUY" else 0.0,
            "exchange": turnover * self.exchange,
            "sebi": turnover * self.sebi,
            "ipft": turnover * self.ipft,
            "dp": dp,
        }
        out["gst"] = self.gst * sum(
            out[k] for k in ["brokerage", "exchange", "sebi", "ipft", "dp"]
        )
        out = {k: v * multiplier for k, v in out.items()}
        out["total"] = sum(out.values())
        return out
