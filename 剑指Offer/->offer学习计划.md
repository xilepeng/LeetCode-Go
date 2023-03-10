


1. [剑指 Offer 09. 用两个栈实现队列](#剑指-offer-09-用两个栈实现队列)
2. [剑指 Offer 30. 包含min函数的栈](#剑指-offer-30-包含min函数的栈)


第 1 天 栈与队列（简单）

## [剑指 Offer 09. 用两个栈实现队列](https://leetcode.cn/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

```go
type CQueue struct {
    inStack,outStack []int 
}

func Constructor() CQueue {
    return CQueue{
    }
}

func (this *CQueue) AppendTail(value int)  {
    this.inStack = append(this.inStack, value)
}

func (this *CQueue) DeleteHead() int {
    if len(this.outStack) == 0 {
        if len(this.inStack) == 0 {
            return -1
        }
        this.in2out()
    }
    head := this.outStack[len(this.outStack)-1]
    this.outStack = this.outStack[:len(this.outStack)-1]
    return head
}

func (this *CQueue) in2out(){
    for len(this.inStack) > 0 {
        this.outStack = append(this.outStack, this.inStack[len(this.inStack)-1])
        this.inStack = this.inStack[:len(this.inStack)-1]
    }
}

/**
 * Your CQueue object will be instantiated and called as such:
 * obj := Constructor();
 * obj.AppendTail(value);
 * param_2 := obj.DeleteHead();
 */
```

## [剑指 Offer 30. 包含min函数的栈](https://leetcode.cn/problems/bao-han-minhan-shu-de-zhan-lcof/)


```go

```