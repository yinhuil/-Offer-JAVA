package leetcodeForJianzhi;
/*题目描述

Sort a linked list using insertion sort.*/
public class insertionSortList {
	static class ListNode{
		ListNode next;
		int val;
		ListNode(int x){
			val=x;
		}
	}
//	解题思路： 这里面current表示当前出现小于前面的数值，然后从前面开始遍历，找到合适的位置进行插入
	public ListNode insertionSortList(ListNode head) {
		if(head==null || head.next==null) return head;
		ListNode root=new ListNode(0);
		root.next=head;
		ListNode pHead=head;
		ListNode current=null;
		ListNode pre=null;
		while(pHead!=null && pHead.next!=null){
			if(pHead.val<=pHead.next.val){
				pHead=pHead.next;
			}else{
				current=pHead.next;
				pHead.next=current.next;
				pre=root;
				while(pre.next.val<=current.val){
					pre=pre.next;
				}
				current.next=pre.next;
				pre.next=current;
			}
		}
		return root.next;
	}
}
