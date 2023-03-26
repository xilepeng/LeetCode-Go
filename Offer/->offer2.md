
1. [13. 找出数组中重复的数字](#13-找出数组中重复的数字)
2. [14. 不修改数组找出重复的数字](#14-不修改数组找出重复的数字)
3. [33. 链表中倒数第k个节点](#33-链表中倒数第k个节点)
4. [70. 二叉搜索树的第k个结点](#70-二叉搜索树的第k个结点)
5. [15. 二维数组中的查找](#15-二维数组中的查找)
6. [46. 二叉搜索树的后序遍历序列](#46-二叉搜索树的后序遍历序列)
7. [28. 在O(1)时间删除链表结点](#28-在o1时间删除链表结点)
8. [53. 最小的k个数](#53-最小的k个数)
9. [36. 合并两个排序的链表](#36-合并两个排序的链表)



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




## [46. 二叉搜索树的后序遍历序列](https://www.acwing.com/problem/content/44/)


```go
func verifySequenceOfBST(postorder []int) bool{
	var dfs func(int, int) bool 
	
	dfs = func(i, j int) bool {
	    if i >= j {
	        return true
	    }
	    left := i 
	    for postorder[left] < postorder[j] {
	        left++
	    }
	    right := left
	    for postorder[right] > postorder[j] {
	        right++
	    }
	    return right == j && dfs(i, right-1) && dfs(right, j-1)
	}
	
	return dfs(0, len(postorder)-1)
}
```



## [28. 在O(1)时间删除链表结点](https://www.acwing.com/problem/content/85/)

```go

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
// 由于是单链表，我们不能找到前驱节点，所以我们不能按常规方法将该节点删除。
// 我们可以换一种思路，将下一个节点的值复制到当前节点，然后将下一个节点删除即可。
func deleteNode(node *ListNode)  {
    node.Val = node.Next.Val
    node.Next = node.Next.Next
}
```



## [53. 最小的k个数](https://www.acwing.com/problem/content/49/)


**2. 结果有序**

```go
func getLeastNumbers(A []int, k int) (res []int) {
	heapSize := len(A)
	buildHeap(A, heapSize)
	for i := heapSize - 1; i >= 0; i-- {
		if k == 0 { // 优化
			break
		}
		res = append(res, A[0])
		k--
		A[0], A[i] = A[i], A[0]
		heapSize--
		minHeapify(A, 0, heapSize)
	}
	return
}

func buildHeap(A []int, heapSize int) {
	for i := heapSize >> 1; i >= 0; i-- {
		minHeapify(A, i, heapSize)
	}
}

// 小根堆：降序
func minHeapify(A []int, i, heapSize int) {
	for i<<1+1 < heapSize {
		lson, rson, small := i<<1+1, i<<1+2, i
		for lson < heapSize && A[lson] < A[small] {
			small = lson
		}
		for rson < heapSize && A[rson] < A[small] {
			small = rson
		}
		if small != i {
			A[i], A[small] = A[small], A[i]
			i = small
		} else {
			break
		}
	}
}
```



## [36. 合并两个排序的链表](https://www.acwing.com/problem/content/description/34/)

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func merge(l1 *ListNode, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    prev := dummy
    for l1 != nil && l2 != nil {
        if l1.Val < l2.Val {
            prev.Next = l1
            l1 = l1.Next
        } else {
            prev.Next = l2
            l2 = l2.Next
        }
        prev = prev.Next
    }
    if l1 == nil {
        prev.Next = l2
    } else {
        prev.Next = l1
    }
    return dummy.Next
}
```



