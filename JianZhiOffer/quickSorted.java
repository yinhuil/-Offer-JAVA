package JianZhiOffer;
//快排 ：  首先以第一个数（low位）为基数base，然后从最后一个开始往前遍历，知道发现一个数high<base，此时将low为放high的数，然后从low开始遍历
//知道发现一个比base数大的数，此时把high位方low位的数据，  一直循环到 low>high，最后将low位放base，返回low位
//最后分别从两边递归
public class quickSorted {
	public static int[] quickSorted(int[] arr,int low ,int high){
		if(high>low){
			int temp=patition(arr,low,high);
			quickSorted(arr, low, temp-1);
			quickSorted(arr, temp+1, high);
		}
		return arr;
	}
	private static int patition(int[] arr, int low, int high) {
		int baseNumber=arr[low];
		while(high>low){
			while(high>low && arr[high]>baseNumber){
				high--;
			}
			arr[low]=arr[high];
			while(high>low && arr[low]<=baseNumber){
				low++;
			}
			arr[high]=arr[low];
		}
		arr[low]=baseNumber;
		return low;
	}
	public static void main(String[] args) {
		// TODO 自动生成的方法存根
		int[] arr = {2,8,7,1,3,5,6,4};
		int[] tag = quickSorted(arr, 0, arr.length-1);
		System.out.println("排序后的结果为:");
		for (int i : tag) {
			System.out.print(i + "\t");
		}
	}
}




























