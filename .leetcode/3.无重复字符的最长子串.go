/*
 * @lc app=leetcode.cn id=3 lang=golang
 *
 * [3] 无重复字符的最长子串
 */

// @lc code=start
func lengthOfLongestSubstring(s string) int {
	start, res := 0, 0
	m := map[byte]int{}
	for i := 0; i < len(s); i++ {
		if _, exists := m[s[i]]; exists {
			start = max(start, m[s[i]]+1)
		}
		m[s[i]] = i
		res = max(res, i-start+1)
	}
	return res
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

// @lc code=end

