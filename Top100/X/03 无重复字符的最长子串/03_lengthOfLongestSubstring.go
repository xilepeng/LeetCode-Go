package main

import (
	"fmt"
)

func lengthOfLongestSubstring(s string) int {
	longest, n := 0, len(s)
	freq := make(map[byte]int, n)
	for i, j := 0, 0; j < n; j++ {
		freq[s[j]]++
		for freq[s[j]] > 1 {
			freq[s[i]]--
			i++
		}
		longest = max(longest, j-i+1) // Go 1.21.1
	}
	return longest
}

func main() {
	s := "abcabcbb"
	res := lengthOfLongestSubstring(s)
	fmt.Println("\"abcabcbb\" 无重复字符的最长子串的长度是：", res)
}
