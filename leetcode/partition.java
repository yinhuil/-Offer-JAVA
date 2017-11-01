package leetcodeForJianzhi;
/*题目描述

Given a linked list and a value x, partition it such that all nodes less 
than x come before nodes greater than or equal to x.
You should preserve the original relative order of the nodes in each of the two partitions.
For example,
Given1->4->3->2->5->2and x = 3,
return1->2->2->4->3->5.*/

//解析，本题是要求大于x的值在list的左边，大于x的值在list的右边，本题用两个list表示，一个表示左边一个表示右边

public class partition {
	static class ListNode {
		ListNode next;
		int val;
		ListNode(int x){
			val=x;
		}
	}
	public ListNode partition(ListNode head, int x){
		ListNode list1=new ListNode(0);
		ListNode list2=new ListNode(0);
		ListNode cur1=list1;
		ListNode cur2=list2;
		while(head!=null){
			if(head.val<x){
				cur1.next=head;
				cur1=cur1.next;
			}else{
				cur2.next=head;
				cur2=cur2.next;
			}
			head=head.next;
		}
		cur2.next=null;
		cur1.next=list2.next;
		return list1.next;
	}
}
