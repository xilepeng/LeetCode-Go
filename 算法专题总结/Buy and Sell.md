
1. [✅ 121. 买卖股票的最佳时机](#-121-买卖股票的最佳时机)
2. [122. 买卖股票的最佳时机 II](#122-买卖股票的最佳时机-ii)

## ✅ [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

``` go
// 最低价格买入，最高价格卖出
func maxProfit(prices []int) int {
	min_price, max_profit := math.MaxInt64, 0
	for _, price := range prices {
		if price < min_price {
			min_price = price // 最低价格
		} 
		if max_profit < price-min_price {
			max_profit = price - min_price // 最高利润
		}
	}
	return max_profit
}
```



``` go
func maxProfit(prices []int) int {
	min_price, max_profit := math.MaxInt64, 0
	for _, price := range prices { // 忘记 _, 导致取到index，而非value 
		max_profit = max(max_profit, price-min_price) // 最低价格
		min_price = min(min_price, price)             // 最高利润
	}
	return max_profit
}
```




``` go
func maxProfit(prices []int) int {
	buy := math.MinInt64 // 买入之后的余额
	sell := 0            // 卖出之后的余额
	for _, p := range prices {
		buy = max(buy, -p) // 无论买/卖，保证手里的钱最多
		sell = max(sell, buy+p)
	}
	return sell
}
```



[参考](https://www.bilibili.com/video/BV1hQ4y1R7pL)




## [122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

- 这一题是第 121 题的加强版。要求输出最大收益，这一题不止买卖一次，可以买卖多次，买卖不能在同一天内操作。
- 最大收益来源，必然是每次跌了就买入，涨到顶峰的时候就抛出。只要有涨峰就开始计算赚的钱，连续涨可以用两两相减累加来计算，两两相减累加，相当于涨到波峰的最大值减去谷底的值。这一点看通以后，题目非常简单。



``` go

func maxProfit(prices []int) int {
	profit := 0
	for i := 1; i < len(prices); i++ {
		if prices[i-1] < prices[i] {
			profit += prices[i] - prices[i-1]
		}
	}
	return profit
}
```

``` go
func maxProfit(prices []int) int {
	profit := 0
	for i := 1; i < len(prices); i++ {
		profit += max(0, prices[i]-prices[i-1])
	}
	return profit
}
```















---




[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

[123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

[188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

[309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

[714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

**思路1**：



------
- I -- General cases

The idea begins with the following question: Given an array representing the price of stocks on each day, what determines the maximum profit we can obtain?

Most of you can quickly come up with answers like "it depends on which day we are and how many transactions we are allowed to complete". Sure, those are important factors as they manifest themselves in the problem descriptions. However, there is a hidden factor that is not so obvious but vital in determining the maximum profit, which is elaborated below.

First let's spell out the notations to streamline our analyses. Let prices be the stock price array with length n, i denote the i-th day (i will go from 0 to n-1), k denote the maximum number of transactions allowed to complete, T[i][k] be the maximum profit that could be gained at the end of the i-th day with at most k transactions. Apparently we have base cases: T[-1][k] = T[i][0] = 0, that is, no stock or no transaction yield no profit (note the first day has i = 0 so i = -1 means no stock). Now if we can somehow relate T[i][k] to its subproblems like T[i-1][k], T[i][k-1], T[i-1][k-1], ..., we will have a working recurrence relation and the problem can be solved recursively. So how do we achieve that?

The most straightforward way would be looking at actions taken on the i-th day. How many options do we have? The answer is three: buy, sell, rest. Which one should we take? The answer is: we don't really know, but to find out which one is easy. We can try each option and then choose the one that maximizes our profit, provided there are no other restrictions. However, we do have an extra restriction saying no multiple transactions are allowed at the same time, meaning if we decide to buy on the i-th day, there should be 0 stock held in our hand before we buy; if we decide to sell on the i-th day, there should be exactly 1 stock held in our hand before we sell. The number of stocks held in our hand is the hidden factor mentioned above that will affect the action on the i-th day and thus affect the maximum profit.

Therefore our definition of T[i][k] should really be split into two: T[i][k][0] and T[i][k][1], where the former denotes the maximum profit at the end of the i-th day with at most k transactions and with 0 stock in our hand AFTER taking the action, while the latter denotes the maximum profit at the end of the i-th day with at most k transactions and with 1 stock in our hand AFTER taking the action. Now the base cases and the recurrence relations can be written as:

	1. Base cases:
	T[-1][k][0] = 0, T[-1][k][1] = -Infinity
	T[i][0][0] = 0, T[i][0][1] = -Infinity

	2. Recurrence relations:
	T[i][k][0] = max(T[i-1][k][0], T[i-1][k][1] + prices[i])
	T[i][k][1] = max(T[i-1][k][1], T[i-1][k-1][0] - prices[i])

For the base cases, T[-1][k][0] = T[i][0][0] = 0 has the same meaning as before while T[-1][k][1] = T[i][0][1] = -Infinity emphasizes the fact that it is impossible for us to have 1 stock in hand if there is no stock available or no transactions are allowed.

For T[i][k][0] in the recurrence relations, the actions taken on the i-th day can only be rest and sell, since we have 0 stock in our hand at the end of the day. T[i-1][k][0] is the maximum profit if action rest is taken, while T[i-1][k][1] + prices[i] is the maximum profit if action sell is taken. Note that the maximum number of allowable transactions remains the same, due to the fact that a transaction consists of two actions coming as a pair -- buy and sell. Only action buy will change the maximum number of transactions allowed (well, there is actually an alternative interpretation, see my comment below).

For T[i][k][1] in the recurrence relations, the actions taken on the i-th day can only be rest and buy, since we have 1 stock in our hand at the end of the day. T[i-1][k][1] is the maximum profit if action rest is taken, while T[i-1][k-1][0] - prices[i] is the maximum profit if action buy is taken. Note that the maximum number of allowable transactions decreases by one, since buying on the i-th day will use one transaction, as explained above.

To find the maximum profit at the end of the last day, we can simply loop through the prices array and update T[i][k][0] and T[i][k][1] according to the recurrence relations above. The final answer will be T[i][k][0] (we always have larger profit if we end up with 0 stock in hand).

------

- II -- Applications to specific cases

The aforementioned six stock problems are classified by the value of k, which is the maximum number of allowable transactions (the last two also have additional requirements such as "cooldown" or "transaction fee"). I will apply the general solution to each of them one by one.






[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

- Case I: k = 1

For this case, we really have two unknown variables on each day: T[i][1][0] and T[i][1][1], and the recurrence relations say:

T[i][1][0] = max(T[i-1][1][0], T[i-1][1][1] + prices[i])
T[i][1][1] = max(T[i-1][1][1], T[i-1][0][0] - prices[i]) = max(T[i-1][1][1], -prices[i])

where we have taken advantage of the base caseT[i][0][0] = 0 for the second equation.

It is straightforward to write the O(n) time and O(n) space solution, based on the two equations above. However, if you notice that the maximum profits on the i-th day actually only depend on those on the (i-1)-th day, the space can be cut down to O(1). Here is the space-optimized solution:



``` go
func maxProfit(prices []int) int {
	T_i10, T_i11 := 0, math.MinInt64
	for _, price := range prices {
		T_i10 = max(T_i10, T_i11+price)
		T_i11 = max(T_i11, 0-price)
	}
	return T_i10
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

Now let's try to gain some insight of the solution above. If we examine the part inside the loop more carefully, T_i11 really just represents the maximum value of the negative of all stock prices up to the i-th day, or equivalently the minimum value of all the stock prices. As for T_i10, we just need to decide which action yields a higher profit, sell or rest. And if action sell is taken, the price at which we bought the stock is T_i11, i.e., the minimum value before the i-th day. This is exactly what we would do in reality if we want to gain maximum profit. I should point out that this is not the only way of solving the problem for this case. You may find some other nice solutions [here](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/discuss/39038/kadanes-algorithm-since-no-one-has-mentioned-about-this-so-far-in-case-if-interviewer-twists-the-input).


[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

- Case II: k = +Infinity

If k is positive infinity, then there isn't really any difference between k and k - 1 (wonder why? see my comment below), which implies T[i-1][k-1][0] = T[i-1][k][0] and T[i-1][k-1][1] = T[i-1][k][1]. Therefore, we still have two unknown variables on each day: T[i][k][0] and T[i][k][1] with k = +Infinity, and the recurrence relations say:

T[i][k][0] = max(T[i-1][k][0], T[i-1][k][1] + prices[i])
T[i][k][1] = max(T[i-1][k][1], T[i-1][k-1][0] - prices[i]) = max(T[i-1][k][1], T[i-1][k][0] - prices[i])

where we have taken advantage of the fact that T[i-1][k-1][0] = T[i-1][k][0] for the second equation. The O(n) time and O(1) space solution is as follows:

``` go
func maxProfit(prices []int) int {
	T_ik0, T_ik1 := 0, math.MinInt64
	for _, price := range prices {
		T_ik0_old := T_ik0
		T_ik0 = max(T_ik0, T_ik1+price)
		T_ik1 = max(T_ik1, T_ik0_old-price)
	}
	return T_ik0
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

``` go
func maxProfit(prices []int) int {
	T_ik0, T_ik1 := 0, math.MinInt64
	for _, price := range prices {
		T_ik1 = max(T_ik1, T_ik0-price)
		T_ik0 = max(T_ik0, T_ik1+price)
	}
	return T_ik0
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

(Note: The caching of the old values of T_ik0, that is, the variable T_ik0_old, is unnecessary. Special thanks to 0x0101 and elvina for clarifying this.)

This solution suggests a greedy strategy of gaining maximum profit: as long as possible, buy stock at each local minimum and sell at the immediately followed local maximum. This is equivalent to finding increasing subarrays in prices (the stock price array), and buying at the beginning price of each subarray while selling at its end price. It's easy to show that this is the same as accumulating profits as long as it is profitable to do so, as demonstrated in this post.


[123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

- Case III: k = 2

Similar to the case where k = 1, except now we have four variables instead of two on each day: T[i][1][0], T[i][1][1], T[i][2][0], T[i][2][1], and the recurrence relations are:

T[i][2][0] = max(T[i-1][2][0], T[i-1][2][1] + prices[i])
T[i][2][1] = max(T[i-1][2][1], T[i-1][1][0] - prices[i])
T[i][1][0] = max(T[i-1][1][0], T[i-1][1][1] + prices[i])
T[i][1][1] = max(T[i-1][1][1], -prices[i])

where again we have taken advantage of the base caseT[i][0][0] = 0 for the last equation. The O(n) time and O(1) space solution is as follows:

``` go
func maxProfit(prices []int) int {
	T_i10, T_i11 := 0, math.MinInt64
	T_i20, T_i21 := 0, math.MinInt64
	for _, price := range prices {
		T_i20 = max(T_i20, T_i21+price)
		T_i21 = max(T_i21, T_i10-price)
		T_i10 = max(T_i10, T_i11+price)
		T_i11 = max(T_i11, -price)
	}
	return T_i20
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

which is essentially the same as the one given [here](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/discuss/39611/is-it-best-solution-with-on-o1).



[188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

- Case IV: k is arbitrary

This is the most general case so on each day we need to update all the maximum profits with different k values corresponding to 0 or 1 stocks in hand at the end of the day. However, there is a minor optimization we can do if k exceeds some critical value, beyond which the maximum profit will no long depend on the number of allowable transactions but instead will be bound by the number of available stocks (length of the prices array). Let's figure out what this critical value will be.

A profitable transaction takes at least two days (buy at one day and sell at the other, provided the buying price is less than the selling price). If the length of the prices array is n, the maximum number of profitable transactions is n/2 (integer division). After that no profitable transaction is possible, which implies the maximum profit will stay the same. Therefore the critical value of k is n/2. If the given k is no less than this value, i.e., k >= n/2, we can extend k to positive infinity and the problem is equivalent to Case II.

The following is the O(kn) time and O(k) space solution. Without the optimization, the code will be met with TLE for large k values.


``` go
func maxProfit(k int, prices []int) int {
	if k >= len(prices)>>1 {
		T_ik0, T_ik1 := 0, math.MinInt64
		for _, price := range prices {
			T_ik0_old := T_ik0
			T_ik0 = max(T_ik0, T_ik1+price)
			T_ik1 = max(T_ik1, T_ik0_old-price)
		}
		return T_ik0
	}
	T_ik0, T_ik1 := make([]int, k+1), make([]int, k+1)
	for i := range T_ik0 {
		T_ik0[i] = 0
		T_ik1[i] = math.MinInt64
	}
	for _, price := range prices {
		for j := k; j > 0; j-- {
			T_ik0[j] = max(T_ik0[j], T_ik1[j]+price)
			T_ik1[j] = max(T_ik1[j], T_ik0[j-1]-price)
		}
	}
	return T_ik0[k]
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

```


[309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

- Case V: k = +Infinity but with cooldown

This case resembles Case II very much due to the fact that they have the same k value, except now the recurrence relations have to be modified slightly to account for the "cooldown" requirement. The original recurrence relations for Case II are given by

T[i][k][0] = max(T[i-1][k][0], T[i-1][k][1] + prices[i])
T[i][k][1] = max(T[i-1][k][1], T[i-1][k][0] - prices[i])

But with "cooldown", we cannot buy on the i-th day if a stock is sold on the (i-1)-th day. Therefore, in the second equation above, instead of T[i-1][k][0], we should actually use T[i-2][k][0] if we intend to buy on the i-th day. Everything else remains the same and the new recurrence relations are

T[i][k][0] = max(T[i-1][k][0], T[i-1][k][1] + prices[i])
T[i][k][1] = max(T[i-1][k][1], T[i-2][k][0] - prices[i])

And here is the O(n) time and O(1) space solution:


``` go
func maxProfit(prices []int) int {
	T_ik0_pre, T_ik0, T_ik1 := 0, 0, math.MinInt64
	for _, price := range prices {
		T_ik0_old := T_ik0
		T_ik0 = max(T_ik0, T_ik1+price)
		T_ik1 = max(T_ik1, T_ik0_pre-price)
		T_ik0_pre = T_ik0_old
	}
	return T_ik0
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```


[714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

- Case VI: k = +Infinity but with transaction fee

Again this case resembles Case II very much as they have the same k value, except now the recurrence relations need to be modified slightly to account for the "transaction fee" requirement. The original recurrence relations for Case II are given by

T[i][k][0] = max(T[i-1][k][0], T[i-1][k][1] + prices[i])
T[i][k][1] = max(T[i-1][k][1], T[i-1][k][0] - prices[i])

Since now we need to pay some fee (denoted as fee) for each transaction made, the profit after buying or selling the stock on the i-th day should be subtracted by this amount, therefore the new recurrence relations will be either

T[i][k][0] = max(T[i-1][k][0], T[i-1][k][1] + prices[i])
T[i][k][1] = max(T[i-1][k][1], T[i-1][k][0] - prices[i] - fee)

or

T[i][k][0] = max(T[i-1][k][0], T[i-1][k][1] + prices[i] - fee)
T[i][k][1] = max(T[i-1][k][1], T[i-1][k][0] - prices[i])

Note we have two options as for when to subtract the fee. This is because (as I mentioned above) each transaction is characterized by two actions coming as a pair - - buy and sell. The fee can be paid either when we buy the stock (corresponds to the first set of equations) or when we sell it (corresponds to the second set of equations). The following are the O(n) time and O(1) space solutions corresponding to these two options, where for the second solution we need to pay attention to possible overflows.

Solution I -- pay the fee when buying the stock:


``` go
func maxProfit(prices []int, fee int) int {
	T_ik0, T_ik1 := 0, math.MinInt64
	for _, price := range prices {
		T_ik0_old := T_ik0
		T_ik0 = max(T_ik0, T_ik1+price)
		T_ik1 = max(T_ik1, T_ik0_old-price-fee)
	}
	return T_ik0
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

``` go
func maxProfit(prices []int, fee int) int {
	T_ik0, T_ik1 := 0, math.MinInt64
	for _, price := range prices {
		T_ik1 = max(T_ik1, T_ik0-price-fee)
		T_ik0 = max(T_ik0, T_ik1+price)
	}
	return T_ik0
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

参考：
[Most consistent ways of dealing with the series of stock problems](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/discuss/108870/Most-consistent-ways-of-dealing-with-the-series-of-stock-problems)


------

**思路2：买卖股票的最佳时机**

**我们要跳出固有的思维模式，并不是要考虑买还是卖，而是要最大化手里持有的钱。
买股票手里的钱减少，卖股票手里的钱增加，无论什么时刻，我们要保证手里的钱最多。
并且我们这一次买还是卖只跟上一次我们卖还是买的状态有关。**

[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)


``` go
// 最低价格买入，最高价格卖出
func maxProfit(prices []int) int {
	min_price, max_profit := math.MaxInt64, 0
	for _, price := range prices {
		if price < min_price {
			min_price = price // 最低价格
		} else if max_profit < price-min_price {
			max_profit = price - min_price // 最高利润
		}
	}
	return max_profit
}
```


``` go
func maxProfit(prices []int) int {
	buy := math.MinInt64 // 买入之后的余额
	sell := 0            // 卖出之后的余额
	for _, p := range prices {
		// 无论买/卖，保证手里的钱最多
		if buy < -p {
			buy = -p
		}
		if sell < buy+p {
			sell = buy + p
		}
	}
	return sell
}
```

``` go
func maxProfit(prices []int) int {
	buy := math.MinInt64 // 买入之后的余额
	sell := 0            // 卖出之后的余额
	for _, p := range prices {
		buy = max(buy, -p) // 无论买/卖，保证手里的钱最多
		sell = max(sell, buy+p)
	}
	return sell
}
```

[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)


``` go
func maxProfit(prices []int) int {
	buy, sell := math.MinInt64, 0
	for _, p := range prices {
		buy = max(buy, sell-p)
		sell = max(sell, buy+p)
	}
	return sell
}
```

这两个问题唯一的不同点在于我们是买一次还是买无穷多次，而代码就只有 0-p 和 sell-p 的区别。
因为如果买无穷多次，就需要上一次卖完的状态。如果只买一次，那么上一个状态一定是0。

[123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)


第三题只允许最多买两次，那么就有四个状态，第一次买，第一次卖，第二次买，第二次卖。
还是那句话，无论什么状态，我们要保证手里的钱最多。

``` go
func maxProfit(prices []int) int {
	b1, b2, s1, s2 := math.MinInt64, math.MinInt64, 0, 0
	for _, p := range prices {
		b1 = max(b1, 0-p)
		s1 = max(s1, b1+p)
		b2 = max(b2, s1-p)
		s2 = max(s2, b2+p)
	}
	return s2
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

[188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

到了第四题，相信大家已经要懂了，第三题最多两次我们有2x2个状态，那么k次我们就需要kx2个状态。
那么我们并不需要像第三题那样真的列kx2个参数，我们只需要两个数组就可以了。
注意，我这里sell后面的[0] 是为了保证第一次买的时候sell[0-1] == 0，
是为了让大家更清楚地看懂迭代的那两句话，当然每个人都有不同的初始化习惯。

python 不懂
```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        k = min(k, len(prices) // 2)
        if k <= 0:
            return 0

        buy = [-float("inf")] * k
        sell = [0] * k + [0]

        for p in prices:
            for j in range(k):
                buy[j] = max(buy[j], sell[j-1] - p)
                sell[j] = max(sell[j], buy[j] + p)

        return sell[-2]

```



[309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

这道题只是第二题的变形，卖完要隔一天才能买，那么就多记录前一天卖的状态即可。

``` go
func maxProfit(prices []int) int {
	buy, sell_pre, sell := math.MinInt64, 0, 0
	for _, p := range prices {
		buy = max(buy, sell_pre-p)
		sell_pre, sell = sell, max(sell, buy+p)
	}
	return sell
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```



[714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

每次买卖需要手续费，那么我们买的时候减掉手续费就行了。
``` go
func maxProfit(prices []int, fee int) int {
	buy, sell := math.MinInt64, 0
	for _, p := range prices {
		buy = max(buy, sell-p-fee)
		sell = max(sell, buy+p)
	}
	return sell
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```









------



**思路3**：

[188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

- 一、穷举框架

利用「状态」进行穷举。我们具体到每一天，看看总共有几种可能的「状态」，再找出每个「状态」对应的「选择」。我们要穷举所有「状态」，穷举的目的是根据对应的「选择」更新状态。听起来抽象，你只要记住「状态」和「选择」两个词就行，下面实操一下就很容易明白了。

``` go
for 状态1 in 状态1的所有取值：
    for 状态2 in 状态2的所有取值：
        for ...
            dp[状态1][状态2][...] = 择优(选择1，选择2...)
```

比如说这个问题，每天都有三种「选择」：买入、卖出、无操作，我们用 buy, sell, rest 表示这三种选择。但问题是，并不是每天都可以任意选择这三种选择的，因为 sell 必须在 buy 之后，buy 必须在 sell 之前。那么 rest 操作还应该分两种状态，一种是 buy 之后的 rest（持有了股票），一种是 sell 之后的 rest（没有持有股票）。而且别忘了，我们还有交易次数 k 的限制，就是说你 buy 还只能在 k > 0 的前提下操作。

很复杂对吧，不要怕，我们现在的目的只是穷举，你有再多的状态，老夫要做的就是一把梭全部列举出来。这个问题的「状态」有三个，第一个是天数，第二个是允许交易的最大次数，第三个是当前的持有状态（即之前说的 rest 的状态，我们不妨用 1 表示持有，0 表示没有持有）。然后我们用一个三维数组就可以装下这几种状态的全部组合：

``` go
dp[i][k][0 or 1]
0 <= i <= n-1, 1 <= k <= K
n 为天数，大 K 为最多交易数
此问题共 n × K × 2 种状态，全部穷举就能搞定。

for 0 <= i < n:
    for 1 <= k <= K:
        for s in {0, 1}:
            dp[i][k][s] = max(buy, sell, rest)
```

而且我们可以用自然语言描述出每一个状态的含义，比如说 dp[3][2][1] 的含义就是：今天是第三天，我现在手上持有着股票，至今最多进行 2 次交易。再比如 dp[2][3][0] 的含义：今天是第二天，我现在手上没有持有股票，至今最多进行 3 次交易。很容易理解，对吧？

我们想求的最终答案是 dp[n - 1][K][0]，即最后一天，最多允许 K 次交易，最多获得多少利润。读者可能问为什么不是 dp[n - 1][K][1]？因为 [1] 代表手上还持有股票，[0] 表示手上的股票已经卖出去了，很显然后者得到的利润一定大于前者。
记住如何解释「状态」，一旦你觉得哪里不好理解，把它翻译成自然语言就容易理解了。

- 二、状态转移框架

现在，我们完成了「状态」的穷举，我们开始思考每种「状态」有哪些「选择」，应该如何更新「状态」。只看「持有状态」，可以画个状态转移图。

![](https://gblobscdn.gitbook.com/assets%2F-MYD_Lf7CsNGNNiGg_9e%2Fsync%2F298b4971971d6e850923f64ab74792b86aa5c668.png?alt=media)

通过这个图可以很清楚地看到，每种状态（0 和 1）是如何转移而来的。根据这个图，我们来写一下状态转移方程：

``` go
dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
              max(   选择 rest  ,             选择 sell      )

解释：今天我没有持有股票，有两种可能：
要么是我昨天就没有持有，然后今天选择 rest，所以我今天还是没有持有；
要么是我昨天持有股票，但是今天我 sell 了，所以我今天没有持有股票了。

dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
              max(   选择 rest  ,           选择 buy         )

解释：今天我持有着股票，有两种可能：
要么我昨天就持有着股票，然后今天选择 rest，所以我今天还持有着股票；
要么我昨天本没有持有，但今天我选择 buy，所以今天我就持有股票了。
```

这个解释应该很清楚了，如果 buy，就要从利润中减去 prices[i]，如果 sell，就要给利润增加 prices[i]。今天的最大利润就是这两种可能选择中较大的那个。而且注意 k 的限制，我们在选择 buy 的时候，把 k 减小了 1，很好理解吧，当然你也可以在 sell 的时候减 1，一样的。

现在，我们已经完成了动态规划中最困难的一步：状态转移方程。如果之前的内容你都可以理解，那么你已经可以秒杀所有问题了，只要套这个框架就行了。不过还差最后一点点，就是定义 base case，即最简单的情况。

``` go
dp[-1][k][0] = 0
解释：因为 i 是从 0 开始的，所以 i = -1 意味着还没有开始，这时候的利润当然是 0 。

dp[-1][k][1] = -infinity
解释：还没开始的时候，是不可能持有股票的，用负无穷表示这种不可能。

dp[i][0][0] = 0
解释：因为 k 是从 1 开始的，所以 k = 0 意味着根本不允许交易，这时候利润当然是 0 。

dp[i][0][1] = -infinity
解释：不允许交易的情况下，是不可能持有股票的，用负无穷表示这种不可能。
```

把上面的状态转移方程总结一下：

``` go
base case：
dp[-1][k][0] = dp[i][0][0] = 0			//没开始或不允许交易时最大利润为 0
dp[-1][k][1] = dp[i][0][1] = -infinity	//没开始或不允许交易时不可能持有股票

状态转移方程：
dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])	  //前一天没有持有或卖出
dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i]) //前一天持有或买入
```

读者可能会问，这个数组索引是 -1 怎么编程表示出来呢，负无穷怎么表示呢？这都是细节问题，有很多方法实现。现在完整的框架已经完成，下面开始具体化。



[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

- 方法三：动态规划

- 第一题，k = 1

直接套状态转移方程，根据 base case，可以做一些化简：

``` go
dp[i][1][0] = max(dp[i-1][1][0], dp[i-1][1][1] + prices[i])
dp[i][1][1] = max(dp[i-1][1][1], dp[i-1][0][0] - prices[i]) 
            = max(dp[i-1][1][1], -prices[i])
解释：k = 0 的 base case，所以 dp[i-1][0][0] = 0。

现在发现 k 都是 1，不会改变，即 k 对状态转移已经没有影响了。
可以进行进一步化简去掉所有 k：
dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
dp[i][1] = max(dp[i-1][1], -prices[i])
```

直接写出代码：

``` go
func maxProfit(prices []int) int {
	n := len(prices)
	dp := make([][]int, n+1)
	for i := 0; i < n; i++ {
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
		dp[i][1] = max(dp[i-1][1], -prices[i])
	}
	return dp[n-1][0]
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

显然 i = 0 时 dp[i-1] 是不合法的。这是因为我们没有对 i 的 base case 进行处理。可以这样处理：

``` go
func maxProfit(prices []int) int {
	n := len(prices)
	dp := make([][]int, n)
	for i := range dp {
		dp[i] = make([]int, 2)
	}
	for i := 0; i < n; i++ {
		if i-1 == -1 {
			dp[i][0] = 0
			//dp[i][0]
			//= max(dp[-1][0], dp[-1][1]+prices[i])
			//= max(0, -1<<63 + prices[i]) = 0
			dp[i][1] = -prices[i]
			//dp[i][1]
			//= max(dp[-1][1], -prices[i])
			//= max(-1<<63, -prices[i])
			//= -prices[i]
			continue
		}
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
		dp[i][1] = max(dp[i-1][1], -prices[i])
	}
	return dp[n-1][0]
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```


第一题就解决了，但是这样处理 base case 很麻烦，而且注意一下状态转移方程，新状态只和相邻的一个状态有关，其实不用整个 dp 数组，只需要一个变量储存相邻的那个状态就足够了，这样可以把空间复杂度降到 O(1):


``` go
func maxProfit(prices []int) int {
	n := len(prices)
	dp_i_0, dp_i_1 := 0, math.MinInt64	//-1<<63
	for i := 0; i < n; i++ {
		// dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
		dp_i_0 = max(dp_i_0, dp_i_1+prices[i])
		// dp[i][1] = max(dp[i-1][1], -prices[i])
		dp_i_1 = max(dp_i_1, -prices[i])
	}
	return dp_i_0
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

``` go
func maxProfit(prices []int) int {
	dp_i_0, dp_i_1, n := 0, math.MinInt64, len(prices)
	for i := 0; i < n; i++ {
		dp_i_0 = max(dp_i_0, dp_i_1+prices[i])
		dp_i_1 = max(dp_i_1, -prices[i])
	}
	return dp_i_0
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```


- 方法一：暴力法

Time Limit Exceeded
201/210 cases passed (N/A)

``` go
func maxProfit(prices []int) int {
	res, n := 0, len(prices)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			res = max(res, prices[j]-prices[i])
		}
	}
	return res
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```



- 方法二：一次遍历

``` go
func maxProfit(prices []int) int {
	minprice, maxprofit := math.MaxInt64, 0
	for _, price := range prices {
		minprice = min(minprice, price)
		maxprofit = max(maxprofit, price-minprice)
	}
	return maxprofit
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
```


``` go
func maxProfit(prices []int) int {
	minprice, maxprofit := math.MaxInt64, 0 // 1<<63-1
	for _, price := range prices {
		if price < minprice {
			minprice = price
		}
		if maxprofit < price-minprice {
			maxprofit = price - minprice
		}
	}
	return maxprofit
}
```

- dp

``` go
func maxProfit(prices []int) int {
	minprice, n := prices[0], len(prices)
	dp := make([]int, n)
	for i, price := range prices {
		if i-1 == -1 {
			dp[i] = 0
			minprice = prices[0]
			continue
		}
		dp[i] = max(dp[i-1], price-minprice)
		minprice = min(minprice, price)
	}
	return dp[n-1]
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
```

- 优化空间

``` go
func maxProfit(prices []int) int {
	minprice, maxprofit := prices[0], 0
	for _, price := range prices {
		maxprofit = max(maxprofit, price-minprice)
		minprice = min(minprice, price)
	}
	return maxprofit
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
```


[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

- 第二题，k = +infinity

如果 k 为正无穷，那么就可以认为 k 和 k - 1 是一样的。可以这样改写框架：

``` go
dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
            = max(dp[i-1][k][1], dp[i-1][k][0] - prices[i])

我们发现数组中的 k 已经不会改变了，也就是说不需要记录 k 这个状态了：
dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
```

直接翻译成代码：

``` go
func maxProfit(prices []int) int {
	dp_i_0, dp_i_1, n := 0, math.MinInt64, len(prices)
	for i := 0; i < n; i++ {
		temp := dp_i_0
		dp_i_0 = max(dp_i_0, dp_i_1+prices[i])
		dp_i_1 = max(dp_i_1, temp-prices[i])
	}
	return dp_i_0
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```


[309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

- 第三题，k = +infinity with cooldown
每次 sell 之后要等一天才能继续交易。只要把这个特点融入上一题的状态转移方程即可：

``` go
dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
dp[i][1] = max(dp[i-1][1], dp[i-2][0] - prices[i])
解释：第 i 天选择 buy 的时候，要从 i-2 的状态转移，而不是 i-1 。
```

``` go
func maxProfit(prices []int) int {
	dp_i_0, dp_i_1, n := 0, math.MinInt64, len(prices)
	dp_pre_0 := 0 //代表 dp[i-2][0]
	for i := 0; i < n; i++ {
		temp := dp_i_0
		dp_i_0 = max(dp_i_0, dp_i_1+prices[i])
		dp_i_1 = max(dp_i_1, dp_pre_0-prices[i])
		dp_pre_0 = temp
	}
	return dp_i_0
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

[714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)


- 第四题，k = +infinity with fee

每次交易要支付手续费，只要把手续费从利润中减去即可。改写方程：

``` go
dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i] - fee)
解释：相当于买入股票的价格升高了。
在第一个式子里减也是一样的，相当于卖出股票的价格减小了。
```

直接翻译成代码：

``` go
func maxProfit(prices []int, fee int) int {
	dp_i_0, dp_i_1, n := 0, math.MinInt64, len(prices)
	for i := 0; i < n; i++ {
		temp := dp_i_0
		dp_i_0 = max(dp_i_0, dp_i_1+prices[i])
		dp_i_1 = max(dp_i_1, temp-prices[i]-fee)
	}
	return dp_i_0
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```




[123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

- 第五题，k = 2

k = 2 和前面题目的情况稍微不同，因为上面的情况都和 k 的关系不太大。要么 k 是正无穷，状态转移和 k 没关系了；要么 k = 1，跟 k = 0 这个 base case 挨得近，最后也没有存在感。
这道题 k = 2 和后面要讲的 k 是任意正整数的情况中，对 k 的处理就凸显出来了。我们直接写代码，边写边分析原因。

``` go
原始的动态转移方程，没有可化简的地方
dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
```


我们必须穷举所有状态。其实我们之前的解法，都在穷举所有状态，只是之前的题目中 k 都被化简掉了。比如说第一题，k = 1：
这道题由于没有消掉 k 的影响，所以必须要对 k 进行穷举：








[188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

- 第六题，k = any integer



