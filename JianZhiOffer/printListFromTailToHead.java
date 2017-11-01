package JianZhiOffer;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class printListFromTailToHead {
//	输入一个链表，从尾到头打印链表每个节点的值。
//	题意是将列表从后往前的输出。   考虑到堆栈的先进后出的特点，就连续的把link list放到堆栈当中 ，最后在放到list里面
	static class ListNode {
		int value;
		ListNode next;

		ListNode(int x) {
			value = x;
		}
	}

	public static ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
		ArrayList<Integer> list = new ArrayList<Integer>();
		Stack<ListNode> sta = new Stack<ListNode>();
		while (listNode != null) {
			sta.push(listNode);
			listNode = listNode.next;
		}
		while (!sta.isEmpty()) {
			list.add(sta.pop().value);
		}
		return list;

	}

	public static void main(String[] args) {
		ListNode node1 = new ListNode(1);
		ListNode node2 = new ListNode(2);
		ListNode node3 = new ListNode(3);
		node1.next = node2;
		node2.next = node3;
		List<Integer> list = printListFromTailToHead(node1);
		for (Integer integer : list) {
			System.out.println(integer);
		}
	}
}
