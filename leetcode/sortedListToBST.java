package leetcodeForJianzhi;
/*题目描述

Given a singly linked list where elements are sorted in ascending order,
convert it to a height balanced BST.*/
//解析：这道题，是首先找到中间的节点，如 1->2->3 就是先找到中间的2，然后左边为1，右边为3， 进行递归；
public class sortedListToBST {
	static class ListNode{
		ListNode next;
		int val;
		ListNode(int x){
			val=x;
		}
	}
	static class TreeNode{
		TreeNode left;
		TreeNode right;
		int val;
		TreeNode(int x){
			val=x;
		}
	}
	public TreeNode sortedListToBST(ListNode head) {
		if(head==null) return null;
		if(head.next==null) return new TreeNode(head.val);
		ListNode fast=head;
		ListNode slow=head;
		ListNode temp=null;
		while(fast!=null && fast.next!=null){
			
			fast=fast.next.next;
			temp=slow;
			slow=slow.next;
		}
		TreeNode pHead=new TreeNode(slow.val);
		if(temp==null){
			head=null;
		}else{
			temp.next=null;
		}
		pHead.left=sortedListToBST(head);
		pHead.right=sortedListToBST(slow.next);
		return pHead;
	}
}
