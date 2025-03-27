def plot_comparison(sid_results, evan_results, location, depth_group, output_path, target_variable, plot_option='both'):
    """Create comparison plot based on specified option (sid, evan, or both)"""
    
    if plot_option == 'both':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        axes = [ax]
    
    def plot_dataset(ax, results, dataset_name):
        ax.plot(results['train_data']['date'], results['train_data']['actual'], 
                'b:', alpha=0.5, label='Train (Actual)', linewidth=1)
        ax.plot(results['train_data']['date'], results['train_data']['predicted'], 
                'b-', alpha=0.5, label='Train (Predicted)')
        ax.plot(results['test_data']['date'], results['test_data']['actual'], 
                'r:', label='Test (Actual)', linewidth=1)
        ax.plot(results['test_data']['date'], results['test_data']['predicted'], 
                'r-', label='Test (Predicted)')
        ax.set_title(f'{dataset_name} Data - {location} at {depth_group}\n(n={results["metrics"]["n_samples"]} samples)')
        ax.legend()
        ax.set_ylabel(target_variable)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    if plot_option in ['both', 'sid']:
        plot_dataset(axes[0], sid_results, 'SID')
    
    if plot_option in ['both', 'evan']:
        plot_idx = 0 if plot_option == 'evan' else 1
        plot_dataset(axes[plot_idx], evan_results, 'EVAN')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()