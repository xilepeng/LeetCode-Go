

[76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/) next

[129. 求根节点到叶节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)

### 方法一：深度优先搜索
思路与算法

从根节点开始，遍历每个节点，如果遇到叶子节点，则将叶子节点对应的数字加到数字之和。如果当前节点不是叶子节点，则计算其子节点对应的数字，然后对子节点递归遍历。

![](https://assets.leetcode-cn.com/solution-static/129/fig1.png)

![](https://pic.leetcode-cn.com/1603933660-UNWQbT-image.png)

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func sumNumbers(root *TreeNode) int {
	var dfs func(*TreeNode, int) int

	dfs = func(root *TreeNode, prevSum int) int {
		if root == nil {
			return 0
		}
		sum := prevSum*10 + root.Val
		if root.Left == nil && root.Right == nil {
			return sum
		}
		return dfs(root.Left, sum) + dfs(root.Right, sum)
	}

	return dfs(root, 0)
}
```

复杂度分析

- 时间复杂度：O(n)，其中 n 是二叉树的节点个数。对每个节点访问一次。

- 空间复杂度：O(n)，其中 n 是二叉树的节点个数。空间复杂度主要取决于递归调用的栈空间，递归栈的深度等于二叉树的高度，最坏情况下，二叉树的高度等于节点个数，空间复杂度为 O(n)。



[958. 二叉树的完全性检验](https://leetcode-cn.com/problems/check-completeness-of-a-binary-tree/)

### 方法一：广度优先搜索
1. 按 根左右(前序遍历) 顺序依次检查
2. 如果出现空节点，标记end = true
3. 如果后面还有节点，返回false

```go
func isCompleteTree(root *TreeNode) bool {
	q, end := []*TreeNode{root}, false
	for len(q) > 0 {
		node := q[0]
		q = q[1:]
		if node == nil {
			end = true
		} else {
			if end == true {
				return false
			}
			q = append(q, node.Left)
			q = append(q, node.Right)
		}
	}
	return true
}
```


[468. 验证IP地址](https://leetcode-cn.com/problems/validate-ip-address/)



[剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

```go
func exchange(nums []int) []int {
    for i, j := 0, 0; i < len(nums); i++ {
        if nums[i] & 1 == 1 { //nums[i]奇数      nums[j]偶数
            nums[i], nums[j] = nums[j], nums[i] //奇偶交换
            j++
        }
    }
    return nums
}
```

```go
func exchange(nums []int) []int {
    i, j := 0, len(nums)-1
    for i < j {
        for i < j && nums[i] & 1 == 1 { //从左往右、奇数一直向右走
            i++
        }
        for i < j && nums[j] & 1 == 0 { //从右向左、偶数一直向左走
            j--
        }
        nums[i], nums[j] = nums[j], nums[i] //奇偶交换
    }
    return nums
}
```


[322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)


![322. Coin Change and 518. Coin Change 2.png](http://ww1.sinaimg.cn/large/007daNw2ly1gps6k2bgrtj31kg3tub29.jpg)

![截屏2021-04-23 16.55.43.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpts6iwvafj319i0o042z.jpg)

![截屏2021-04-23 13.16.57.png](http://ww1.sinaimg.cn/large/007daNw2ly1gptltwipl8j319q0p2go8.jpg)


#### iterate amount

```go
func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	dp[0] = 0 //base case
	for i := 1; i < len(dp); i++ {
		dp[i] = amount + 1
	}
	for i := 1; i <= amount; i++ { //遍历所有状态的所有值
		for _, coin := range coins { //求所有选择的最小值 min(dp[4],dp[3],dp[0])+1
			if i-coin >= 0 {
				dp[i] = min(dp[i], dp[i-coin]+1)
			}
		}
	}
	if amount-dp[amount] < 0 {
		return -1
	}
	return dp[amount]
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
```



[518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

![截屏2021-04-23 16.57.11.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpts6y27nhj319a0n8gpb.jpg)

#### iterate coins

```go
func change(amount int, coins []int) int {
	dp := make([]int, amount+1)
	dp[0] = 1
	for _, coin := range coins {
		for i := coin; i <= amount; i++ {
			dp[i] += dp[i-coin]
		}
	}
	return dp[amount]
}
```




[剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)


```go
type CQueue struct {
    inStack, outStack []int
}

func Constructor() CQueue {
    return CQueue{}
}

func (this *CQueue) AppendTail(value int)  {
    this.inStack = append(this.inStack, value)
}

func (this *CQueue) DeleteHead() int {
    if len(this.outStack) == 0 {
        if len(this.inStack) == 0 { return -1}
        for len(this.inStack) > 0 {
            top := this.inStack[len(this.inStack)-1]
            this.inStack = this.inStack[:len(this.inStack)-1]
            this.outStack = append(this.outStack, top)
        }
    }
    top := this.outStack[len(this.outStack)-1]
    this.outStack = this.outStack[:len(this.outStack)-1]
    return top
}

/**
 * Your CQueue object will be instantiated and called as such:
 * obj := Constructor();
 * obj.AppendTail(value);
 * param_2 := obj.DeleteHead();
 */
```



[162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)

### 方法一：二分查找

```go
func findPeakElement(nums []int) int {
	left, right := 0, len(nums)-1
	for left < right {
		mid := left + (right-left)>>1
		if nums[mid] > nums[mid+1] {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return left
}
```
### 方法二: 线性扫描
```go
func findPeakElement(nums []int) int {
	for i := 0; i < len(nums)-1; i++ {
		if nums[i] > nums[i+1] {
			return i
		}
	}
	return len(nums) - 1
}
```


[179. 最大数](https://leetcode-cn.com/problems/largest-number/)



[188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

### 一、穷举框架

利用「状态」进行穷举。我们具体到每一天，看看总共有几种可能的「状态」，再找出每个「状态」对应的「选择」。我们要穷举所有「状态」，穷举的目的是根据对应的「选择」更新状态。听起来抽象，你只要记住「状态」和「选择」两个词就行，下面实操一下就很容易明白了。

```go
for 状态1 in 状态1的所有取值：
    for 状态2 in 状态2的所有取值：
        for ...
            dp[状态1][状态2][...] = 择优(选择1，选择2...)
```

比如说这个问题，每天都有三种「选择」：买入、卖出、无操作，我们用 buy, sell, rest 表示这三种选择。但问题是，并不是每天都可以任意选择这三种选择的，因为 sell 必须在 buy 之后，buy 必须在 sell 之前。那么 rest 操作还应该分两种状态，一种是 buy 之后的 rest（持有了股票），一种是 sell 之后的 rest（没有持有股票）。而且别忘了，我们还有交易次数 k 的限制，就是说你 buy 还只能在 k > 0 的前提下操作。

很复杂对吧，不要怕，我们现在的目的只是穷举，你有再多的状态，老夫要做的就是一把梭全部列举出来。这个问题的「状态」有三个，第一个是天数，第二个是允许交易的最大次数，第三个是当前的持有状态（即之前说的 rest 的状态，我们不妨用 1 表示持有，0 表示没有持有）。然后我们用一个三维数组就可以装下这几种状态的全部组合：

```go
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

### 二、状态转移框架

现在，我们完成了「状态」的穷举，我们开始思考每种「状态」有哪些「选择」，应该如何更新「状态」。只看「持有状态」，可以画个状态转移图。

![](https://gblobscdn.gitbook.com/assets%2F-MYD_Lf7CsNGNNiGg_9e%2Fsync%2F298b4971971d6e850923f64ab74792b86aa5c668.png?alt=media)

通过这个图可以很清楚地看到，每种状态（0 和 1）是如何转移而来的。根据这个图，我们来写一下状态转移方程：

```go
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

```go
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

```go
base case：
dp[-1][k][0] = dp[i][0][0] = 0			//没开始或不允许交易时最大利润为 0
dp[-1][k][1] = dp[i][0][1] = -infinity	//没开始或不允许交易时不可能持有股票

状态转移方程：
dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])	  //前一天没有持有或卖出
dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i]) //前一天持有或买入
```

读者可能会问，这个数组索引是 -1 怎么编程表示出来呢，负无穷怎么表示呢？这都是细节问题，有很多方法实现。现在完整的框架已经完成，下面开始具体化。



[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

### 方法三：动态规划

#### 第一题，k = 1

直接套状态转移方程，根据 base case，可以做一些化简：

```go
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

```go
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

```go
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


```go
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


### 方法一：暴力法

Time Limit Exceeded
201/210 cases passed (N/A)

```go
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



### 方法二：一次遍历

```go
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


```go
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

### dp

```go
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

### 优化空间

```go
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

#### 第二题，k = +infinity

如果 k 为正无穷，那么就可以认为 k 和 k - 1 是一样的。可以这样改写框架：

```go
dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
            = max(dp[i-1][k][1], dp[i-1][k][0] - prices[i])

我们发现数组中的 k 已经不会改变了，也就是说不需要记录 k 这个状态了：
dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
```

直接翻译成代码：

```go
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


[123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)






[309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)



[714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)







[7. 整数反转](https://leetcode-cn.com/problems/reverse-integer/)





[460. LFU 缓存](https://leetcode-cn.com/problems/lfu-cache/)




[补充题6. 手撕堆排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)



[59. 螺旋矩阵 II](https://leetcode-cn.com/problems/spiral-matrix-ii/)



[128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)



[剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

```go
func fib(n int) int {
    if n < 2 { 
        return n 
    }
    prev, curr := 0, 1
    for i := 2; i <= n; i++ {
        sum := prev + curr
        prev = curr
        curr = sum % 1000000007
    }
    return curr 
}
```


[509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/)

### 方法一：递归

![截屏2021-04-22 11.34.30.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpsd9a3yaxj317o0o0q7h.jpg)

```go
func fib(n int) int {
	if n == 0 || n == 1 {
		return n
	}
	return fib(n-1) + fib(n-2)
}
```

复杂度分析

- 时间复杂度：O(2^n)。
- 空间复杂度：O(h)。

### 方法二：带备忘录递归


![截屏2021-04-23 20.07.36.png](http://ww1.sinaimg.cn/large/007daNw2ly1gptxon19akj319k0nw407.jpg)

闭包写法：
```go
func fib(n int) int {
	memo := make([]int, n+1)//从0开始
	var helper func(int) int

	helper = func(n int) int {
		if n == 0 || n == 1 {
			return n
		}
		if memo[n] != 0 {
			return memo[n]
		}
		memo[n] = helper(n-1) + helper(n-2)
		return memo[n]
	}

	return helper(n)
}
```

```go
func fib(n int) int {
	memo := make([]int, n+1)
	return helper(memo, n)
}
func helper(memo []int, n int) int {
	if n < 2 {
		return n
	}
	if memo[n] != 0 { //剪枝
		return memo[n]
	}
	memo[n] = helper(memo, n-1) + helper(memo, n-2)
	return memo[n]
}
```

复杂度分析

- 时间复杂度：O(n)。
- 空间复杂度：O(n)。

### 方法三：动态规划

![截屏2021-04-22 11.35.07.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpsda7wdjwj30zu0hwmyn.jpg)

```go
func fib(n int) int {
	if n == 0 {
		return 0
	}
	dp := make([]int, n+1)
	dp[0], dp[1] = 0, 1       //base case
	for i := 2; i <= n; i++ { //状态转移
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}
```
复杂度分析

- 时间复杂度：O(n)。
- 空间复杂度：O(n)。

### 方法四：滚动数组

![截屏2021-04-22 11.42.22.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpsdgjxvorj31520kgjsx.jpg)

动态规划空间优化：只存储前2项

```go
func fib(n int) int {
	if n == 0 || n == 1 { //base case
		return n
	} //递推关系
	prev, curr := 0, 1
	for i := 2; i <= n; i++ {
		next := prev + curr
		prev = curr
		curr = next
	}
	return curr
}
```

复杂度分析

- 时间复杂度：O(n)。
- 空间复杂度：O(1)。

[24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)



[32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)



[498. 对角线遍历](https://leetcode-cn.com/problems/diagonal-traverse/)