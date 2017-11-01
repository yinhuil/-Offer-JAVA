package JianZhiOffer;


public class FindKthToTai15 {
	
//	输入一个链表，输出该链表中倒数第k个结点。
	
//	解析：  首先将链表往前移动k-1步，然后在同时往后移
	static class ListNode {
		int val;
		ListNode next;

		ListNode(int x) {
			val = x;
		}
	}

	public ListNode FindKthToTail(ListNode head, int k) {
		if (head == null || k <= 0)
			return null;
		ListNode pre = head;
		ListNode last = head;
		for (int i = 1; i < k; i++) {
			if (pre.next != null) {
				pre = pre.next;
			} else {
				return null;
			}
		}
		while (pre.next != null) {
			pre = pre.next;
			last = last.next;
		}
		return last;

	}
}
