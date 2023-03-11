
1. [13. 找出数组中重复的数字](#13-找出数组中重复的数字)
2. [14. 不修改数组找出重复的数字](#14-不修改数组找出重复的数字)
3. [33. 链表中倒数第k个节点](#33-链表中倒数第k个节点)
4. [70. 二叉搜索树的第k个结点](#70-二叉搜索树的第k个结点)
5. [15. 二维数组中的查找](#15-二维数组中的查找)



## [13. 找出数组中重复的数字](https://www.acwing.com/problem/content/14/)

```go
func duplicateInArray(nums []int) int {
    n := len(nums)
    for _,x := range nums {
        if x < 0 || x >= n {
            return -1
        }
    }
    for i := range nums {
        for nums[i] != nums[nums[i]] {
            nums[i], nums[nums[i]] = nums[nums[i]], nums[i]
        }
        if nums[i] != i {
            return nums[i]
        }
    }
    return -1
}
    
```

## [14. 不修改数组找出重复的数字](https://www.acwing.com/problem/content/15/)

```go
func duplicateInArray(nums []int) int {
    l, r := 1, len(nums)-1
    for l < r {
        mid := (l+r)>>1 // 划分的区间：[l, mid], [mid + 1, r]
        s := 0
        for _,x := range nums {
            if x >= l && x <= mid {
                s++
            }
        }
        if s > mid-l+1 { // [l,mid] 区间个数大于区间长度，有重复
            r = mid
        } else {
            l = mid+1
        }
    }
    return r
}
```

## [33. 链表中倒数第k个节点](https://www.acwing.com/problem/content/32/)

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func findKthToTail(head *ListNode, k int) *ListNode {
    p :=  head
    for i := 0; i < k; i++{
        if p == nil {
            return nil
        }
        p = p.Next
    }
    slow, fast := head, head
    for fast != nil {
        if k > 0 {
            fast = fast.Next
            k--
        } else {
            slow, fast = slow.Next, fast.Next
        }
    }
    return slow
}

func findKthToTail1(head *ListNode, k int) *ListNode {
    n := 0 
    for p := head; p != nil; p = p.Next{
        n++
    }
    if n < k {
        return nil
    }
    p := head
    for i := 0; i < n-k; i++ {
        p = p.Next
    }
    return p
}
```



## [70. 二叉搜索树的第k个结点](https://www.acwing.com/problem/content/66/)

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func kthNode(root *TreeNode, k int) (res *TreeNode) {
    var dfs func(root *TreeNode) 
    
    dfs = func(root *TreeNode) {
        if root == nil {
            return 
        }
        dfs(root.Left)
        k--
        if k == 0 {
            res = root
        }
        dfs(root.Right)
    }
    
    dfs(root)
    return 
}
```



## [15. 二维数组中的查找](https://www.acwing.com/problem/content/16/)

```go
func searchArray(matrix [][]int, target int) bool {
    if len(matrix) == 0 {
        return false
    }
    x, y := 0, len(matrix[0])-1
    for x < len(matrix) && y >= 0 {
        if matrix[x][y] == target {
            return true
        }
        if target < matrix[x][y] {
            y--
        } else {
            x++
        }
    }
    return false
}
```