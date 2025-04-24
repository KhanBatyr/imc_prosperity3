import json
import math
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from statistics import NormalDist
from typing import Any, TypeAlias

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

def BS_CALL(S, K, T, r, sigma):
    "S - stock price, K - strike price, T - time to expiration, r - risk-free rate, sigma - volatility"
    N = NormalDist().cdf
    d1 = (math.log(S/K) + (r + sigma**2/2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * N(d1) - K * math.exp(-r*T)* N(d2)

import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

PARAMS = {
    Product.VOLCANIC_ROCK: {
        "default_edge": 1,
        "soft_edge": 1,
        "hard_edge": 2,
        "reversion_beta": -0.41,
        "alpha": 0.66,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "default_edge": 1,
        "soft_edge": 0.015,
        "hard_edge": 0.03,
        "reversion_beta": -0.35,
        "alpha": 0.86,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "default_edge": 1,
        "soft_edge": 0.02,
        "hard_edge": 0.04,
        "reversion_beta": -0.35,
        "alpha": 0.6,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "default_edge": 1,
        "soft_edge": 0.0005,
        "hard_edge": 0.001,
        "reversion_beta": -0.5,
        # "alpha": 0.0089,
        "alpha": 0.001,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "default_edge": 1,
        "soft_edge": 0.0005,
        "hard_edge": 0.001,
        "reversion_beta": -0.35,
        "alpha": 0.2,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "default_edge": 1,
        "soft_edge": 0.0005,
        "hard_edge": 0.001,
        "reversion_beta": -0.35,
        "alpha": 0.35,
    },
}

class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        # TODO: Add logic

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0

        self.act(state)

        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class SignalStrategy(Strategy):
    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2

    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.sell_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position

        self.buy(price, to_buy)

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = max(order_depth.buy_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_sell = self.limit + position

        self.sell(price, to_sell)


class CoconutStrategy(SignalStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.threshold = None

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position
        cur_price = self.get_mid_price(state, "VOLCANIC_ROCK")

        # Update the McGinley Dynamic Moving Average
        # If not set, initialize it to the current price.
        if not hasattr(self, "long_run_mean"):
            self.long_run_mean = cur_price
        else:
            self.long_run_mean = (1 - PARAMS[self.symbol]['alpha']) * self.long_run_mean + PARAMS[self.symbol]['alpha'] * cur_price



        moving_avg = self.long_run_mean
        deviation = cur_price - moving_avg
        true_value = cur_price + PARAMS[self.symbol]["reversion_beta"] * deviation

        max_buy_price = true_value - PARAMS[self.symbol]["hard_edge"] if position > self.limit * 0.5 else true_value - PARAMS[self.symbol]["soft_edge"]
        min_sell_price = true_value + PARAMS[self.symbol]["hard_edge"] if position < -self.limit * 0.5 else true_value + PARAMS[self.symbol]["soft_edge"]

        # Process sell orders (i.e. buy from the market)
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity
                position += quantity

        if to_buy > 0 and buy_orders:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(int(max_buy_price), popular_buy_price + PARAMS[self.symbol]["default_edge"])
            self.buy(price, to_buy)
            to_buy -= to_buy
            position += to_buy

        # Process buy orders (i.e. sell into the market)
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity
                position -= quantity

        if to_sell > 0 and sell_orders:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(int(min_sell_price), popular_sell_price - PARAMS[self.symbol]["default_edge"])
            self.sell(price, to_sell)
            to_sell -= to_sell
            position -= to_sell











    def save(self) -> JSON:
        return self.threshold

    def load(self, data: JSON) -> None:
        self.threshold = data

def brentq(f, a, b, args=(), xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100, full_output=False, disp=False):
    if not isinstance(args, (list, tuple)):
        args = (args,)
    
    fa = f(a, *args)
    fb = f(b, *args)
    if fa * fb > 0:
        msg = "The function must have different signs at a and b."
        if disp:
            raise ValueError(msg)
        else:
            return None
    c = a
    fc = fa
    d = e = b - a
    eps = 2.220446049250313e-16
    for iter in range(maxiter):
        if fb * fc > 0:
            c = a
            fc = fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = rtol * abs(b) + xtol * 0.5
        m = 0.5 * (c - b)
        if abs(m) <= tol1 or fb == 0:
            if full_output:
                return b, (a, c, fa, fb, fc, d, e, iter+1)
            return b
        # Decide if we can use interpolation
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = b - fb * (b - a) / (fb - fa)
            # Conditions on s to accept interpolation
            if not ((3 * a + b) / 4 <= s <= b) or (abs(s - b) >= abs(m) / 2):
                s = b + m
                e = d = m
            else:
                d = s - b
        else:
            s = b + m
            e = d = m
        fs = f(s, *args)
        a, fa = b, fb
        if fs * fb < 0:
            c, fc = s, fs
        else:
            b, fb = s, fs
    if disp:
        raise RuntimeError("Maximum number of iterations exceeded.")
    else:
        if full_output:
            return b, (a, c, fa, fb, fc, d, e, maxiter)
        return b

class CoconutCouponStrategy(SignalStrategy):


    def act(self, state: TradingState) -> None:
        if self.symbol == "VOLCANIC_ROCK_VOUCHER_9500":
            K = 9500
        
        if self.symbol == "VOLCANIC_ROCK_VOUCHER_9750":
            K = 9750

        if self.symbol == "VOLCANIC_ROCK_VOUCHER_10000":
            K = 10000

        if self.symbol == "VOLCANIC_ROCK_VOUCHER_10250":
            K = 10250

        if self.symbol == "VOLCANIC_ROCK_VOUCHER_10500":
            K = 10500

        if "VOLCANIC_ROCK" not in state.order_depths or len(state.order_depths["VOLCANIC_ROCK"].buy_orders) == 0 or len(state.order_depths["VOLCANIC_ROCK"].sell_orders) == 0:
            return

        if self.symbol not in state.order_depths or len(state.order_depths[self.symbol].buy_orders) == 0 or len(state.order_depths[self.symbol].sell_orders) == 0:
            return

        coco = self.get_mid_price(state, "VOLCANIC_ROCK")
        coup = self.get_mid_price(state, self.symbol)

        S = coco
        T = (8 * 1e6 - state.timestamp) / 1e6 / 365
        r = 0

        def objective_function(sigma):
            return BS_CALL(S, K, T, r, sigma) - coup

        implied_vol = brentq(objective_function, -1e-6, 5.0)

        if implied_vol:

            if not hasattr(self, "long_run_mean"):
                self.long_run_mean = implied_vol
            else:
                self.long_run_mean = (1 - PARAMS[self.symbol]['alpha']) * self.long_run_mean + PARAMS[self.symbol]['alpha'] * implied_vol

            # if implied_vol > self.long_run_mean + PARAMS[self.symbol]['default_edge']:
            #     self.go_short(state)
            # elif implied_vol < self.long_run_mean - PARAMS[self.symbol]['default_edge']:
            #     self.go_long(state)

        deviation = implied_vol - self.long_run_mean

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        position = state.position.get(self.symbol, 0)

        to_buy = self.limit - position
        to_sell = self.limit + position

        # Calculate the true volatility based on the deviation
        true_volat = implied_vol + PARAMS[self.symbol]["reversion_beta"] * deviation

        max_buy_volat = true_volat - PARAMS[self.symbol]["hard_edge"] if position > self.limit * 0.5 else true_volat - PARAMS[self.symbol]["soft_edge"]
        min_sell_volat= true_volat + PARAMS[self.symbol]["hard_edge"] if position < -self.limit * 0.5 else true_volat + PARAMS[self.symbol]["soft_edge"]

        # Process sell orders (i.e. buy from the market)
        for price, volume in sell_orders:
            
            coup = price
            ord_implied_vol = brentq(objective_function, -1e-6, 5.0)

            if to_buy > 0 and ord_implied_vol <= max_buy_volat: #if implied volat < max_buy_volat, we can buy:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity
                position += quantity

        if to_buy > 0 and buy_orders:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(math.floor(BS_CALL(S, K, T, r, max_buy_volat)), popular_buy_price + PARAMS[self.symbol]["default_edge"]) #maybe should try math.floor
            self.buy(price, to_buy)
            to_buy -= to_buy
            position += to_buy

        # Process buy orders (i.e. sell into the market)
        for price, volume in buy_orders:
            coup = price
            ord_implied_vol = brentq(objective_function, -1e-6, 5.0)

            if to_sell > 0 and ord_implied_vol >= min_sell_volat:  #if implied volat > min_sell_volat:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity
                position -= quantity

        if to_sell > 0 and sell_orders:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(int(BS_CALL(S, K, T, r, min_sell_volat)), popular_sell_price - PARAMS[self.symbol]["default_edge"]) #math.floor
            self.sell(price, to_sell)
            to_sell -= to_sell
            position -= to_sell


class Trader:
    def __init__(self) -> None:
        limits = {

            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }

        self.strategies: dict[Symbol, Strategy] = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {

            "VOLCANIC_ROCK": CoconutStrategy,
            "VOLCANIC_ROCK_VOUCHER_9500": CoconutCouponStrategy,
            "VOLCANIC_ROCK_VOUCHER_9750": CoconutCouponStrategy,
            "VOLCANIC_ROCK_VOUCHER_10000": CoconutCouponStrategy,
            "VOLCANIC_ROCK_VOUCHER_10250": CoconutCouponStrategy,
            "VOLCANIC_ROCK_VOUCHER_10500": CoconutCouponStrategy,
        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            if symbol in state.order_depths:
                strategy_orders, strategy_conversions = strategy.run(state)
                orders[symbol] = strategy_orders
                conversions += strategy_conversions

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data