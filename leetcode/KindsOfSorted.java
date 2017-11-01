package leetcodeForJianzhi;

public class KindsOfSorted {
//选择排序
	public static void chosedSorted(int [] arr){
		int lens=arr.length;
		for(int i=0;i<lens;i++){
			int start=arr[i];
			int index=i;
			for(int j=i+1;j<lens;j++){
				if(arr[j]<start){
					start=arr[j];
					index=j;
				}
			}
			int temp=arr[i];
			arr[i]=start;
			arr[index]=temp;
		}
	}
	//插入排序
	public static void insertSorted(int[] arr){
		int lens=arr.length;
		if(lens==1 || lens==0) return;
		for(int i=1;i<lens;i++){
			for(int j=i;j>0;j--){
				if(arr[j-1]>arr[j]){
					int temp=arr[j-1];
					arr[j-1]=arr[j];
					arr[j]=temp;
				}else{
					break;
				}
			}
		}
	}
//	归并排序
	public static void mergeSorted(int[] arr){
		if(arr==null || arr.length<2) return ;
		merge_sorted(arr,0,arr.length-1);
	}
	private static void merge_sorted(int[] arr, int left, int right) {
		if(left>=right) return;
		merge_sorted(arr, left, (left+right)/2);
		merge_sorted(arr, (left+right)/2+1, right);
		merge_array(arr,left,right);
	
	}
	private static void merge_array(int[] arr, int left, int right) {
		int mid=(left+right)/2;
		int l1=left;
		int l2=mid+1;
		int[] temp=new int[right-left+1];
		int index=0;
		while(l1<=mid && l2<=right){
			if(arr[l1]<=arr[l2]){
				temp[index++]=arr[l1++];
			}else{
				temp[index++]=arr[l2++];
			}
		}
		while(l1<=mid){
			temp[index++]=arr[l1++];
		}
		while(l2<=right){
			temp[index++]=arr[l2++];
		}
		for(int i=0;i<(right-left+1);i++){
			arr[left+i]=temp[i];
		}
	}
//	快排
	public static void quickSorted(int[] arr, int low ,int high){
		if (high > low) {
			int mid = quick_sort(arr, low, high);
			quickSorted(arr, low, mid);
			quickSorted(arr, mid + 1, high);
		}
	}
	private static int quick_sort(int[] arr, int low, int high) {
		int temp=arr[low];
		while(high>low){
			while(high>low && arr[high]>temp){
				high--;
			}
			arr[low]=arr[high];
			while(high>low && arr[low]<=temp){
				low++;
			}
			arr[high]=arr[low];
		}
		arr[low]=temp;
		return low;
	}
//	希尔排序
	public static void shellSorted(int[] arr){
		int n=arr.length-1;
		int step=n/2;
		for(int i=step;i>0;i--){
			process(arr,i,n);
		}
	}

	private static void process(int[] arr, int step, int len) {
		for(int i=step;i<len;i++){
			int left=i-step;
			int right=i;
			while(left>=0){
				if(arr[left]>arr[right]){
					int temp=arr[left];
					arr[left]=arr[right];
					arr[right]=temp;
					
					left-=step;
					right-=step;
				}else{
					break;
				}
			}
		}
	}
	
	public static void main(String[] args) {
		int[] in ={4,3,2,2,6,7,8};
//		chosedSorted(in);
//		insertSorted(in);
		mergeSorted(in);
//		quickSorted(in, 0, 5);
//		shellSorted(in);
		for (int i : in) {
			System.out.println(i);
		}
	}
}
