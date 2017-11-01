package JianZhiOffer;

import java.util.Stack;

public class minStack21 {
	static Stack<Integer> data=new Stack<Integer>();
	static Stack<Integer> min=new Stack<Integer>();
	public static void push(int value){
		data.push(value);
		if(min.isEmpty() || value<min.peek()){
			min.push(value);
		}else{
			min.push(min.peek());
		}
	}
	public static void pop(){
		data.pop();
		min.pop();
	}
	public static int  min(){
		return min.peek();
	}
	public static void main(String[] args) {
		push(3);
		push(4);
		push(2);
		push(1);
		System.out.println(min());
		pop();
		System.out.println(min());
	}
}
