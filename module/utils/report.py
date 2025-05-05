#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Report generation module

This module provides the functionality to generate model reports, including the ability to draw ROC curves, confusion matrices, regression results, and other visualization charts,
and convert reports to Word and PDF formats.
"""

import os
import shutil
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pypandoc
import dtreeviz

from sklearn.metrics import roc_curve, auc, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import label_binarize

from module.autofe.utils import is_combination_feature, name2formula, split_num_cat_features

# Configure matplotlib to display Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def markdown_to_word(input_file: str, output_file: str) -> None:
    """
    Convert Markdown file to Word document
    
    Args:
        input_file: Input Markdown file path
        output_file: Output Word file path
    """
    pypandoc.convert_file(input_file, 'docx', format='md', outputfile=output_file)


def markdown_to_pdf(input_file: str, output_file: str) -> None:
    """
    Convert Markdown file to PDF document
    
    Args:
        input_file: Input Markdown file path
        output_file: Output PDF file path
    """
    template = os.path.join(os.getcwd(), 'doc', 'pm-template.latex')
    os.system(f'pandoc -s {input_file} -o {output_file} --pdf-engine=xelatex --template={template}')


def mkdirs_save_path(path: str) -> str:
    """
    Create directory and return path
    
    If the directory part of the path does not exist, create the directory
    
    Args:
        path: File path
        
    Returns:
        Original path
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path

def draw_and_save(model: Any, model_type: str, task_type: str, X_train: pd.DataFrame, 
              y_train: Union[pd.DataFrame, pd.Series, np.ndarray], features: List[str], 
              target_name: str, save_path: str, val_result: Dict[str, Any]) -> None:
    """
    Draw and save report charts to the specified path

    Generate different visualization charts based on model type and task type, including tree structure, feature importance, feature correlation, etc., and save to the specified path.

    Args:
        model: Model object, can be sklearn, lightgbm, xgboost or catboost model
        model_type: Model type, such as 'lgbm', 'xgboost', etc.
        task_type: Task type, 'classification' or 'regression'
        X_train: Training set feature data
        y_train: Training set target variable
        features: Feature name list
        target_name: Target variable name
        save_path: Save path
        val_result: Validation set result dictionary
    """

    # Initialize variables
    adv_feature_count = 0
    processed_features = []
    drop_content = []
    insteal_content = []
    
    # Create feature table in Markdown format
    markdown_table = '| AdvFeat Num         | Feature name      |\n|-------------|------------|\n'
    
    # Process feature names, convert combination features to AdvFeat format
    for i, feature in enumerate(features):
        if is_combination_feature(feature):
            formula = name2formula(feature)
            name = f'${formula}$'
            adv_feature_id = f'AdvFeat{adv_feature_count + 1}'
            markdown_table += f'| ${adv_feature_id}$        | {name}       |\n'
            processed_features.append(adv_feature_id)
            adv_feature_count += 1
        else:
            processed_features.append(feature)
    
    # Update feature list and training data column names
    features = processed_features
    X_train_subset = X_train[features].copy()
    X_train_subset.columns = features
    
    # Initialize tree visualization flag
    can_plot_tree = False
    
    # Process tree model visualization
    if model_type in ['lgbm']:
        try:
            # Get the correct model object
            if model_type == 'lgbm':
                try:
                    tree_model = model.model.estimator._Booster
                except AttributeError:
                    tree_model = model
            else:
                tree_model = model.model.estimator
                
            # Create visualization model
            viz_model = dtreeviz.model(
                tree_model, 
                tree_index=1, 
                X_train=X_train_subset, 
                y_train=y_train, 
                feature_names=features, 
                target_name=target_name
            )
            
            # Save different views of the tree structure
            # Standard view
            v = viz_model.view()
            _path = mkdirs_save_path(os.path.join(save_path, 'tree_structure', 'tree_structure1.svg'))
            v.save(_path)
            
            # Horizontal view
            v = viz_model.view(orientation='LR')
            _path = mkdirs_save_path(os.path.join(save_path, 'tree_structure', 'tree_structure2.svg'))
            v.save(_path)
            
            # Simple view
            v = viz_model.view(fancy=False)
            _path = mkdirs_save_path(os.path.join(save_path, 'tree_structure', 'tree_structure3.svg'))
            v.save(_path)
            
            # Limit depth view
            v = viz_model.view(depth_range_to_display=(1, 2))
            _path = mkdirs_save_path(os.path.join(save_path, 'tree_structure', 'tree_structure4.svg'))
            v.save(_path)
            
            # Single sample prediction path
            if len(X_train_subset) > 10:
                sample_idx = 10
                v = viz_model.view(x=X_train_subset.iloc[sample_idx])
                _path = mkdirs_save_path(os.path.join(save_path, 'prediction_path_explanations', 'prediction_path1.svg'))
                v.save(_path)
                
                # Only display path prediction view
                v = viz_model.view(x=X_train_subset.iloc[sample_idx], show_just_path=True)
                _path = mkdirs_save_path(os.path.join(save_path, 'prediction_path_explanations', 'prediction_path2.svg'))
                v.save(_path)
            
            # Leaf node size distribution
            fig, ax = plt.subplots()
            viz_model.leaf_sizes(ax=ax)
            _path = mkdirs_save_path(os.path.join(save_path, 'leaf_info', 'leaf_info1.png'))
            fig.savefig(_path)
            plt.close(fig)
            
            # Leaf node distribution
            fig, ax = plt.subplots()
            viz_model.ctree_leaf_distributions(ax=ax)
            _path = mkdirs_save_path(os.path.join(save_path, 'leaf_info', 'leaf_info2.png'))
            fig.savefig(_path)
            plt.close(fig)
            
            # Tree visualization successful
            can_plot_tree = True
            
        except Exception as e:
            # Record error but continue execution
            can_plot_tree = False
            print(f"Tree visualization failed: {str(e)}")
            logger.error(f"Tree visualization failed: {str(e)}")

    # Get feature importance
    try:
        # Try using feature_importances_ attribute
        feature_importance = pd.DataFrame(
            zip(model.feature_importances_, features), 
            columns=['Value', 'Feature']
        )
    except AttributeError:
        try:
            # Try using feature_importance method
            feature_importance = pd.DataFrame(
                zip(model.feature_importance(), features), 
                columns=['Value', 'Feature']
            )
        except Exception as e:
            # If both methods fail, create an empty DataFrame
            print(f"Error getting feature importance: {str(e)}")
            feature_importance = pd.DataFrame(columns=['Value', 'Feature'])
            # To ensure subsequent code runs normally, add some default data
            if features:
                feature_importance = pd.DataFrame(
                    zip([1.0] * len(features), features),
                    columns=['Value', 'Feature']
                )
    
    # If it is a tree model, draw the feature importance chart
    if is_tree_model(model_type) and not feature_importance.empty:
        # Standardize feature importance as a percentage
        max_value = feature_importance['Value'].max()
        if max_value > 0:  # Prevent division by zero error
            feature_importance['Value'] = 100 * (feature_importance['Value'] / max_value)
        
        # Sort by importance in descending order
        feature_importance = feature_importance.sort_values(
            by='Value', ascending=False
        ).reset_index(drop=True)
        
        # Adjust chart size based on number of features
        if len(feature_importance) > 90:
            plt.figure(figsize=(12, 16))
        elif len(feature_importance) > 60:
            plt.figure(figsize=(12, 13))
        elif len(feature_importance) > 30:
            plt.figure(figsize=(11, 10))
        else:
            plt.figure(figsize=(10, 6))
        
        # Draw bar chart, showing up to 100 features
        sns.barplot(
            x='Value', 
            y='Feature', 
            data=feature_importance.head(100),
            palette='viridis'
        )
        
        # Set chart title and labels
        plt.title('Feature Importance Sorting', fontsize=14)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature Name', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save chart
        importance_path = mkdirs_save_path(
            os.path.join(save_path, 'feature_importance', 'feature_importance1.png')
        )
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()  # Use close while plt.clf() releases resources more completely
    # Create feature importance table
    markdown_table2 = '| Feature name         | Feature Importance      |\n|-------------|------------|\n'
    
    # Add importance scores for combination features
    for feature, score in zip(feature_importance['Feature'].values, feature_importance['Value'].values):
        if feature.startswith('AdvFeat'):
            markdown_table2 += f'| ${feature}$        | ${score:.2f}$       |\n'
    
    # Merge tables
    if markdown_table is not None:
        markdown_table += '\n'
        markdown_table += markdown_table2
    else:
        markdown_table = markdown_table2
    
    # Draw correlation heatmap
    try:
        # Select the top 10 most important features
        top_features = feature_importance['Feature'].values[:10]
        
        # Ensure there are enough features
        if len(top_features) > 1:
            # Calculate correlation matrix
            correlation_data = pd.concat([
                X_train_subset[top_features].reset_index(drop=True), 
                pd.Series(y_train, name=target_name).reset_index(drop=True)
            ], axis=1)
            
            correlation_matrix = correlation_data.corr()
            
            # Draw heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                fmt='.2f', 
                linewidths=0.5
            )
            plt.title('Correlation Heatmap', fontsize=14)
            plt.tight_layout()
            
            # Save chart
            heatmap_path = mkdirs_save_path(
                os.path.join(save_path, 'feature_space', 'feature_space1.png')
            )
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("Not enough features to draw correlation heatmap")
    except Exception as e:
        print(f"Error drawing correlation heatmap: {str(e)}")
        logger.error(f"Error drawing correlation heatmap: {str(e)}")

    # Select numeric and categorical features for visualization
    try:
        # Split features into categorical and numeric features
        cat_features, num_features = split_num_cat_features(X_train_subset)
        
        # Initialize selected features
        numeric_feature = None
        categorical_feature = None
        
        # Select a numeric feature and a categorical feature from important features
        for feature in feature_importance['Feature'].values:
            # Skip combination features
            if not feature.startswith('AdvFeat'):
                if feature in cat_features and categorical_feature is None:
                    categorical_feature = feature
                elif feature not in cat_features and numeric_feature is None:
                    numeric_feature = feature
                
                # If both types of features are found, stop the loop
                if numeric_feature and categorical_feature:
                    break
        
        # If no numeric feature is found, try again
        if numeric_feature is None:
            for feature in feature_importance['Feature'].values:
                if feature not in cat_features:
                    numeric_feature = feature
                    break
        
        # If still no numeric feature is found, use the first feature
        if numeric_feature is None and len(feature_importance['Feature']) > 0:
            numeric_feature = feature_importance['Feature'].values[0]
        
        # If no categorical feature is found, try again
        if categorical_feature is None:
            for feature in feature_importance['Feature'].values:
                if feature in cat_features:
                    categorical_feature = feature
                    break
        
        # Draw the kernel density plot of the numeric feature
        if numeric_feature is not None:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(X_train_subset[numeric_feature], shade=True, color='skyblue')
            plt.title(f'{numeric_feature} Kernel Density Plot', fontsize=14)
            plt.xlabel('Value', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.tight_layout()
            
            # Save chart
            kde_path = mkdirs_save_path(
                os.path.join(save_path, 'feature_space', 'feature_space2.png')
            )
            plt.savefig(kde_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Draw the box plot of the categorical feature
        if categorical_feature is not None:
            plt.figure(figsize=(10, 6))
            
            # Prepare data
            box_data = pd.concat([
                X_train_subset.reset_index(drop=True), 
                pd.Series(y_train, name=target_name).reset_index(drop=True)
            ], axis=1)
            
            # Draw the box plot
            sns.boxplot(
                x=categorical_feature, 
                y=target_name, 
                data=box_data, 
                palette='Set3'
            )
            plt.title(f'{categorical_feature}与{target_name}的箱线图', fontsize=14)
            plt.tight_layout()
            
            # Save chart
            box_path = mkdirs_save_path(
                os.path.join(save_path, 'feature_space', 'feature_space3.png')
            )
            plt.savefig(box_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            # If there is no categorical feature, add to the list of content to be replaced
            drop_content.append('![img](./config/images/feature_space/feature_space3.png)')
            insteal_content.append('**⚠Your box plot did not generate normally! If the dataset does not have a categorical feature, it may cause the image to fail to generate!**')
        
        # Draw trend comparison chart
        if numeric_feature is not None:
            # Create save path
            trends_path = mkdirs_save_path(
                os.path.join(save_path, 'feature_space', 'feature_space4.png')
            )
            
            # Draw trend chart
            plot_trends(
                X_train_subset[numeric_feature].to_numpy(), 
                y_train, 
                label1=numeric_feature, 
                label2=target_name, 
                save_path=trends_path
            )
    except Exception as e:
        print(f"Error drawing feature visualization chart: {str(e)}")
        logger.error(f"Error drawing feature visualization chart: {str(e)}")

    # Generate report
    try:
        # Get report root path
        report_path = os.path.dirname(save_path)
        
        # Select report template based on task type and model type
        if task_type == 'classification':
            # Select classification report template
            if can_plot_tree:
                md_name = 'ESPML分类任务报告.md'
            else:
                md_name = 'ESPML分类报告.md'
            
            # Prepare classification metrics
            ori_content = [
                '该任务的准确率为: ', 
                '该任务的召回率为: ', 
                '该任务的精确率为: ', 
                '该任务的f1为: '
            ]
            
            # Extract metrics from validation results
            try:
                acc = val_result.get('val_Accuracy', 'N/A')
                recall = val_result.get('val_Recall', 'N/A')
                precision = val_result.get('val_Precision', 'N/A')
                f1 = val_result.get('val_f1', 'N/A')
                
                # Format metric values
                rep_content = [
                    f'该任务的准确率为: {acc:.4f}' if isinstance(acc, float) else f'该任务的准确率为: {acc}',
                    f'该任务的召回率为: {recall:.4f}' if isinstance(recall, float) else f'该任务的召回率为: {recall}',
                    f'该任务的精确率为: {precision:.4f}' if isinstance(precision, float) else f'该任务的精确率为: {precision}',
                    f'该任务的f1为: {f1:.4f}' if isinstance(f1, float) else f'该任务的f1为: {f1}'
                ]
            except Exception as e:
                print(f"提取分类指标时出错: {str(e)}")
                rep_content = [
                    '该任务的准确率为: N/A', 
                    '该任务的召回率为: N/A', 
                    '该任务的精确率为: N/A', 
                    '该任务的f1为: N/A'
                ]
        else:
            # Select regression report template
            if can_plot_tree:
                md_name = 'ESPML回归任务报告.md'
            else:
                md_name = 'ESPML回归报告.md'
            
            # Regression task currently has no content to be replaced
            ori_content = []
            rep_content = []
            
            # If you need to add regression metrics, you can expand here
        
        # Build report file path
        md_path = f'doc/{md_name}'
        report_name = md_path.split('/')
        md_to_path = os.path.join(report_path, report_name[-1])
        
        # Copy report template
        try:
            shutil.copyfile(md_path, md_to_path)
        except FileNotFoundError:
            print(f"找不到报告模板文件: {md_path}")
            # 创建一个简单的报告模板
            with open(md_to_path, 'w', encoding='utf-8') as f:
                f.write(f"# ESPML {task_type} 任务报告\n\n")
                f.write("## 模型性能\n\n")
                for content in rep_content:
                    f.write(f"{content}\n")
                f.write("\n## 特征重要性\n\n")
                f.write("![img](./config/images/feature_importance/feature_importance1.png)\n\n")
                f.write("## 特征可视化\n\n")
                f.write("![img](./config/images/feature_space/feature_space1.png)\n\n")
                f.write("![img](./config/images/feature_space/feature_space2.png)\n\n")
                f.write("![img](./config/images/feature_space/feature_space3.png)\n\n")
                f.write("![img](./config/images/feature_space/feature_space4.png)\n\n")
        
        # Update image paths and content in the report
        update_image_path(
            md_to_path, 
            save_path, 
            add_content=markdown_table, 
            replace_content=[ori_content + drop_content, rep_content + insteal_content]
        )
        
        # Generate Word and PDF reports
        output_file_name = md_name.split('.')[0]
        try:
            # Generate Word document
            markdown_to_word(
                md_to_path, 
                os.path.join(report_path, f'{output_file_name}.docx')
            )
            
            # Generate PDF document
            markdown_to_pdf(
                md_to_path, 
                os.path.join(report_path, f'{output_file_name}.pdf')
            )
        except Exception as e:
            print(f"生成Word或PDF报告时出错: {str(e)}")
        
        # Restore image paths in the report
        update_image_path(md_to_path, save_path, reverse=True)
        
    except Exception as e:
        print(f"生成报告时出错: {str(e)}")


def update_image_path(markdown_file_path: str, save_path: str, reverse: bool = False, 
                   add_content: Optional[str] = None, 
                   replace_content: List[List[str]] = [[], []]) -> None:
    """
    Update image paths in the Markdown file
    
    Replace relative paths with absolute paths in the Markdown file, or vice versa.
    Additional content can also be added and specified text can be replaced.
    
    Args:
        markdown_file_path: Markdown file path
        save_path: Image save path
        reverse: Whether to reverse the replacement (replace absolute paths with relative paths)
        add_content: Content to add to the end of the file
        replace_content: Content to replace, format as [original content list, new content list]
    """
    try:
        # Read Markdown file
        with open(markdown_file_path, 'r', encoding='utf-8') as file:
            markdown_content = file.read()
        
        # Replace specified content
        for i in range(len(replace_content[0])):
            if i < len(replace_content[1]):
                markdown_content = markdown_content.replace(replace_content[0][i], replace_content[1][i])
        
        # Define image path mapping
        old_image_paths = [
            './config/images/tree_structure/tree_structure1.svg',
            './config/images/tree_structure/tree_structure2.svg',
            './config/images/tree_structure/tree_structure3.svg',
            './config/images/tree_structure/tree_structure4.svg',
            './config/images/prediction_path_explanations/prediction_path1.svg',
            './config/images/prediction_path_explanations/prediction_path2.svg',
            './config/images/leaf_info/leaf_info1.png',
            './config/images/leaf_info/leaf_info2.png',
            './config/images/feature_importance/feature_importance1.png',
            './config/images/feature_space/feature_space1.png',
            './config/images/feature_space/feature_space2.png',
            './config/images/feature_space/feature_space3.png',
            './config/images/feature_space/feature_space4.png',
            './config/images/valid_metirc/valid_metirc1.png',
            './config/images/valid_metirc/valid_metirc2.png'
        ]
        
        new_image_paths = [
            os.path.join(save_path, 'tree_structure', 'tree_structure1.svg'),
            os.path.join(save_path, 'tree_structure', 'tree_structure2.svg'),
            os.path.join(save_path, 'tree_structure', 'tree_structure3.svg'),
            os.path.join(save_path, 'tree_structure', 'tree_structure4.svg'),
            os.path.join(save_path, 'prediction_path_explanations', 'prediction_path1.svg'),
            os.path.join(save_path, 'prediction_path_explanations', 'prediction_path2.svg'),
            os.path.join(save_path, 'leaf_info', 'leaf_info1.png'),
            os.path.join(save_path, 'leaf_info', 'leaf_info2.png'),
            os.path.join(save_path, 'feature_importance', 'feature_importance1.png'),
            os.path.join(save_path, 'feature_space', 'feature_space1.png'),
            os.path.join(save_path, 'feature_space', 'feature_space2.png'),
            os.path.join(save_path, 'feature_space', 'feature_space3.png'),
            os.path.join(save_path, 'feature_space', 'feature_space4.png'),
            os.path.join(save_path, 'valid_metirc', 'valid_metirc1.png'),
            os.path.join(save_path, 'valid_metirc', 'valid_metirc2.png')
        ]
        
        # Add additional content
        if add_content is not None:
            markdown_content += add_content
        
        # Update image paths
        if not reverse:
            # Replace relative paths with absolute paths
            for i in range(min(len(old_image_paths), len(new_image_paths))):
                markdown_content = markdown_content.replace(old_image_paths[i], new_image_paths[i])
        else:
            # Replace absolute paths with relative paths
            for i in range(min(len(old_image_paths), len(new_image_paths))):
                markdown_content = markdown_content.replace(new_image_paths[i], old_image_paths[i])
        
        # Write back to file
        with open(markdown_file_path, 'w', encoding='utf-8') as file:
            file.write(markdown_content)
            
    except Exception as e:
        print(f"Error updating image paths: {str(e)}")
        # logger.error(f"更新图片路径时出错: {str(e)}")


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, n_classes: int, path: str) -> None:
    """
    Draw ROC curve

    Draw ROC (Receiver Operating Characteristic) curve for classification model prediction results,
    including ROC curves for each class, micro average ROC curve, and macro average ROC curve.
    
    Args:
        y_true: True label array
        y_score: Array of predicted probabilities for each class for each sample
        n_classes: Number of classes
        path: Path to save the chart
    """
    try:
        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Initialize dictionary to store FPR, TPR, and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Handle multi-class and binary classification cases
        if n_classes > 2:
            # Multi-class case
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Calculate micro average ROC curve (treat all samples and classes as binary classification)
            fpr['micro'], tpr['micro'], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
            roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
        else:
            # Binary classification case
            n_classes = 1  # Actually only one class ROC curve
            
            # Ensure array shape is correct
            if y_score.shape[1] >= 2:
                # Use the second column (positive class probability)
                fpr[0], tpr[0], _ = roc_curve(y_true_bin[:, 0], y_score[:, 1])
                roc_auc[0] = auc(fpr[0], tpr[0])
                fpr['micro'], tpr['micro'], _ = roc_curve(y_true_bin.ravel(), y_score[:, 1].ravel())
            else:
                # When there is only one column, use it directly
                fpr[0], tpr[0], _ = roc_curve(y_true_bin[:, 0], y_score[:, 0])
                roc_auc[0] = auc(fpr[0], tpr[0])
                fpr['micro'], tpr['micro'], _ = roc_curve(y_true_bin.ravel(), y_score[:, 0].ravel())
            
            roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
        
        # Calculate macro average ROC curve (average of all class ROC curves)
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
        mean_tpr /= n_classes
        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
        
        # Create chart
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
        
        # Draw ROC curve for each class
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i], tpr[i], 
                color=color, 
                lw=2, 
                label=f'ROC curve - class {i} (area = {roc_auc[i]:0.2f})'
            )
        
        # Draw micro average ROC curve
        plt.plot(
            fpr['micro'], tpr['micro'], 
            label=f'Micro average ROC curve (area = {roc_auc["micro"]:0.2f})', 
            color='deeppink', 
            linestyle=':', 
            linewidth=4
        )
        
        # Draw macro average ROC curve
        plt.plot(
            fpr['macro'], tpr['macro'], 
            label=f'Macro average ROC curve (area = {roc_auc["macro"]:0.2f})', 
            color='navy', 
            linestyle=':', 
            linewidth=4
        )
        
        # Draw random guess line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Set chart properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC curve (Receiver Operating Characteristic)', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the chart to release resources
        
    except Exception as e:
        print(f"Error drawing ROC curve: {str(e)}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: List[int], 
                       normalize: bool = False, title: Optional[str] = None, 
                       cmap: plt.cm = plt.cm.Blues, path: Optional[str] = None) -> None:
    """
    Draw confusion matrix

    Draw confusion matrix for classification model prediction results
    this function is based on sklearn.metrics.confusion_matrix

    Args:
        y_true: True label array
        y_pred: Predicted label array
        classes: List of class labels
        normalize: Whether to normalize the confusion matrix, default is False
        title: Chart title, default is generated based on normalize parameter
        cmap: color map, default is Blues
        path: save the chart path, if None, show the chart without saving
    """
    try:
        # Set chart title
        if not title:
            if normalize:
                title = 'Normalized Confusion Matrix'
            else:
                title = 'Confusion Matrix'
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # If need to normalize, divide each row by the row sum
        if normalize:
            with np.errstate(divide='ignore', invalid='ignore'):
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                # Handle possible NaN values (when row sum is 0)
                cm = np.nan_to_num(cm, nan=0.0)
        
        # Create chart, size adjusted dynamically based on number of classes
        plt.figure(figsize=(max(8, len(classes) + 2), max(6, len(classes) + 2)))
        
        # Show confusion matrix
        im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=14)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        # Set axis ticks
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha='right')
        plt.yticks(tick_marks, classes)
        
        # Set value format and color
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.0
        
        # Show values in the confusion matrix
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(
                    j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=10 if len(classes) > 10 else 12
                )
        
        # Adjust layout and labels
        plt.tight_layout()
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Save or display the chart
        if path:
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()  # Close the chart to release resources
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error drawing confusion matrix: {str(e)}")


def plot_regression_results(y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str = 'Model', save_path: Optional[str] = None) -> None:
    """
    Draw regression model prediction results chart

    Draw scatter plot of predicted values and true values for regression tasks,
    and display model evaluation metrics including mean squared error (MSE) and R^2.
    
    Args:
        y_true: True value array
        y_pred: Predicted value array
        model_name: Model name, used for chart title and legend
        save_path: Save chart path, if None, display chart without saving
    """
    try:
        # Calculate evaluation metrics
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Create chart
        plt.figure(figsize=(10, 6))
        
        # 绘制散点图
        plt.scatter(
            y_true, y_pred, 
            color='blue', 
            edgecolors=(0, 0, 0), 
            alpha=0.7,
            label=f'{model_name} prediction'
        )
        
        # 绘制理想线（对角线）
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot(
            [min_val, max_val], 
            [min_val, max_val], 
            'k--', 
            lw=2,
            label='Ideal line'
        )
        
        # Set chart title and labels
        plt.title(f'{model_name} regression results', fontsize=14)
        plt.xlabel('True value', fontsize=12)
        plt.ylabel('Predicted value', fontsize=12)
        plt.legend(loc='upper left', fontsize=10)
        
        # Add grid and evaluation metric text
        plt.grid(True, alpha=0.3)
        plt.text(
            min_val, 
            max_val, 
            f'MSE = {mse:.4f}\nR^2 = {r2:.4f}', 
            fontsize=12, 
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the chart
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close the chart to release resources
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error drawing regression results chart: {str(e)}")
        # 可以添加日志记录
        # logger.error(f"绘制回归结果图时出错: {str(e)}")


def plot_trends(array1: np.ndarray, array2: np.ndarray, label1: str = 'Array 1', 
               label2: str = 'Array 2', save_path: Optional[str] = None) -> None:
    """
    绘制两个数组的变化趋势对比图

    将两个数组归一化到不同的范围并绘制对比图，以直观地显示它们的变化趋势。
    
    Args:
        array1: 第一个数组
        array2: 第二个数组
        label1: 第一个数组的标签
        label2: 第二个数组的标签
        save_path: 保存图表的路径，如果为None则显示图表而不保存
    """
    try:
        # 限制数组长度并归一化
        max_samples = min(len(array1), len(array2), 100)  # 最多显示100个样本
        
        # 归一化数组到不同范围
        norm_array1 = normalize_array(array1[:max_samples], 1, 2)
        norm_array2 = normalize_array(array2[:max_samples], 0, 1)
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 创建样本ID数组
        x = range(1, len(norm_array1) + 1)
        
        # 绘制两个数组的趋势线
        plt.plot(x, norm_array1, label=label1, linewidth=2, marker='o', markersize=4)
        plt.plot(x, norm_array2, label=label2, linewidth=2, marker='s', markersize=4)
        
        # Add horizontal reference line
        plt.axhline(y=1, color='black', linestyle='--', linewidth=2)
        
        # Set chart title and labels
        plt.title('Feature and label normalized trend comparison', fontsize=14)
        plt.xlabel('Sample ID', fontsize=12)
        plt.ylabel('Normalized value', fontsize=12)
        
        # Add legend and grid
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the chart
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close the chart to release resources
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error drawing trend comparison chart: {str(e)}")
        # 可以添加日志记录
        # logger.error(f"绘制趋势对比图时出错: {str(e)}")


def normalize_array(array: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
    """
    Use min-max normalization method to scale the value range of the array to [new_min, new_max].
    
    Args:
        array: Array to be normalized
        new_min: Minimum value after normalization
        new_max: Maximum value after normalization
        
    Returns:
        Array after normalization
    """
    try:
        # 处理空数组
        if len(array) == 0:
            return np.array([])
            
        # 计算原始数组的最小值和最大值
        min_val = np.min(array)
        max_val = np.max(array)
        
        # 处理数组中所有值相同的情况
        if min_val == max_val:
            return np.full_like(array, (new_min + new_max) / 2)
        
        # 执行最小-最大归一化
        normalized_array = new_min + (new_max - new_min) * (array - min_val) / (max_val - min_val)
        return normalized_array
        
    except Exception as e:
        print(f"Error normalizing array: {str(e)}")
        # 可以添加日志记录
        # logger.error(f"数组归一化时出错: {str(e)}")
        
        # 返回原始数组
        return array
def is_tree_model(model_type: str) -> bool:
    """
    Determine if the model type is a tree model
    
    Check if the given model type belongs to a tree-based model (e.g., LightGBM, XGBoost, CatBoost, etc.).
    
    Args:
        model_type: Model type string
        
    Returns:
        True if the model is a tree model, otherwise False
    """
    tree_models = [
        'lgbm',          # LightGBM
        'xgboost',       # XGBoost
        'xgb_limitdepth', # XGBoost with limited depth
        'catboost',      # CatBoost
        'rf',            # Random Forest
        'extra_tree',    # Extra Trees
        'decision_tree'  # Decision Tree
    ]
    return model_type.lower() in tree_models


if __name__ == '__main__':
    """
    Main function of the test script
    
    When running this script directly, it executes a simple test to convert a Markdown file to a Word document.
    """
    try:
        # Set input and output file paths
        input_markdown = 'doc/ESPML分类任务报告.md'
        output_word = 'output.docx'
        
        # Convert Markdown to Word
        markdown_to_word(input_markdown, output_word)
        print(f"Successfully converted {input_markdown} to {output_word}")
        
    except Exception as e:
        print(f"Conversion failed: {str(e)}")