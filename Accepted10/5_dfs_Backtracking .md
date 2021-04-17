# DFS and BackTracking 

[17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

[79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

[46. 全排列](https://leetcode-cn.com/problems/permutations/)

[47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

[78. 子集](https://leetcode-cn.com/problems/subsets/)

[90. 子集 II](https://leetcode-cn.com/problems/subsets-ii/)

[216. 组合总和 III](https://leetcode-cn.com/problems/combination-sum-iii/)

[52. N皇后 II](https://leetcode-cn.com/problems/n-queens-ii/)

[37. 解数独](https://leetcode-cn.com/problems/sudoku-solver/)

[473. 火柴拼正方形](https://leetcode-cn.com/problems/matchsticks-to-square/)

------

## 题解

[17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

```go 
func letterCombinations(digits string) []string {
	if len(digits) == 0 {
		return []string{}
	}
	//数字字母映射
	mp := map[string]string{
		"2": "abc",
		"3": "def",
		"4": "ghi",
		"5": "jkl",
		"6": "mno",
		"7": "pqrs",
		"8": "tuv",
		"9": "wxyz",
	}
	var ans []string
	var dfs func(int, string)

	// DFS 函数定义
	dfs = func(i int, path string) {
		if i >= len(digits) {
			ans = append(ans, path)
			return
		}

		for _, c := range mp[string(digits[i])] {
			dfs(i+1, path+string(c))
		}
	}
	//执行回溯函数
	dfs(0, "")
	return ans
}
```



[79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

[46. 全排列](https://leetcode-cn.com/problems/permutations/)

[47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

[78. 子集](https://leetcode-cn.com/problems/subsets/)

[90. 子集 II](https://leetcode-cn.com/problems/subsets-ii/)

[216. 组合总和 III](https://leetcode-cn.com/problems/combination-sum-iii/)

[52. N皇后 II](https://leetcode-cn.com/problems/n-queens-ii/)

[37. 解数独](https://leetcode-cn.com/problems/sudoku-solver/)

[473. 火柴拼正方形](https://leetcode-cn.com/problems/matchsticks-to-square/)



