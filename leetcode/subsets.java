package leetcodeForJianzhi;

import java.util.ArrayList;
import java.util.Arrays;

/*题目描述

Given a set of distinct integers, S, return all possible subsets.
Note:
Elements in a subset must be in non-descending order.
The solution set must not contain duplicate subsets.

For example,
If S =[1,2,3], a solution is:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]*/
public class subsets {
    public static  ArrayList<ArrayList<Integer>> subsets(int[] S) {
        ArrayList<ArrayList<Integer>> result=new ArrayList<ArrayList<Integer>>();
        Arrays.sort(S);
        ArrayList<Integer> list=new ArrayList<Integer>();
        for(int i=1;i<=S.length;i++){
        	helper(result,list,S,i,0);
        }
        return result;
    }

	private static void helper(ArrayList<ArrayList<Integer>> result,
			ArrayList<Integer> list, int[] s, int k, int start) {
		if(k<0) return;
		else if(k==0){
			result.add(new ArrayList<Integer>(list));
		}else{
			for(int i=start;i<s.length;i++){
				list.add(s[i]);
				helper(result, list, s, k-1, i+1);
				list.remove(list.size()-1);
			}
		}
		
	}
	public static void main(String[] args) {
		int[] in={1,2};
		ArrayList<ArrayList<Integer>> list=subsets(in);
		for (ArrayList<Integer> arrayList : list) {
			System.out.println(arrayList.toString());
		}
	}
}
