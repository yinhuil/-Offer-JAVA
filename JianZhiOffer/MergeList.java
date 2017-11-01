package JianZhiOffer;


public class MergeList {
	
//	输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则
	
//	本题采用递归的方式，如果当先节点发现小的话，就赋值给pMerge.next,
	static class ListNode {
		int val;
		ListNode next;

		ListNode(int x) {
			val = x;
		}
	}

	public ListNode Merge(ListNode list1, ListNode list2) {
		if (list1 == null)
			return list2;
		else if (list2 == null)
			return list1;
		ListNode pMerge = null;
		if (list1.val < list2.val) {
			pMerge = list1;
			pMerge.next = Merge(list1.next, list2);
		} else {
			pMerge = list2;
			pMerge.next = Merge(list1, list2.next);
		}
		return pMerge;

	}
}
