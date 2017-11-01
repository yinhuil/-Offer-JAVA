package leetcodeForJianzhi;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class subsetsWithDup {
/*	题目描述

	Given a collection of integers that might contain duplicates, S, return all possible subsets.
	Note:
	Elements in a subset must be in non-descending order.
	The solution set must not contain duplicate subsets.

	For example,
	If S =[1,2,2], a solution is:
	[
	  [2],
	  [1],
	  [1,2,2],
	  [2,2],
	  [1,2],
	  []
	]*/
	
//	深度优先的一个经典案例
	public static ArrayList<ArrayList<Integer>> subsetsWithDup(int[] num) {
		ArrayList<ArrayList<Integer>> result=new ArrayList<ArrayList<Integer>>();
		if(num.length==0) return result;
		ArrayList<Integer> list=new ArrayList<Integer>();
		Arrays.sort(num);
		dfs(num,result,list,0);
		return result;
	}

	private static void dfs(int[] num, ArrayList<ArrayList<Integer>> result,
			ArrayList<Integer> list, int index) {
		result.add(new ArrayList<Integer>(list));
		for(int i=index;i<num.length;i++){
			if(i>index && num[i]==num[i-1]) continue;
			list.add(num[i]);
			dfs(num, result, list, i+1);
			list.remove(list.size()-1);
		}
	}
	public static void main(String[] args) {
		int[] in={1,2,3};
		ArrayList<ArrayList<Integer>> list=subsetsWithDup(in);
		for (ArrayList<Integer> arrayList : list) {
			System.out.println(arrayList.toString());
		}
		
	}
}
