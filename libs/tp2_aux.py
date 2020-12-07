#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions for assignment 2
"""
import numpy as np
from skimage.io import imread

def images_as_matrix(N=563):
    """
    Reads all N images in the images folder (indexed 0 through N-1)
    returns a 2D numpy array with one image per row and one pixel per column
    """
    return np.array([imread(f'images/{ix}.png',as_gray=True).ravel() for ix in range(563)])
        

def report_clusters(ids, labels, report_file):
    """Generates html with cluster report
    ids is a 1D array with the id numbers of the images in the images/ folder
    labels is a 1D array with the corresponding cluster labels
    """
    diff_lbls = list(np.unique(labels))
    diff_lbls.sort()
    html = ["""<!DOCTYPE html>
    <html lang="en">
       <head>
        <meta http-equiv="content-type" content="text/html; charset=utf-8">
        <meta charset="UTF-8">
        <title>Cluster Report</title>
       </head>
       <body>
       """]
    for lbl in diff_lbls:
        html.append(f"<h1>Cluster {lbl}</h1>")        
        lbl_imgs = ids[labels==lbl]          
        for count,img in enumerate(lbl_imgs):                
            html.append(f'<img src="images/{int(img)}.png" />')
            #if count % 10 == 9:
            #    html.append('<br/>')
    html.append("</body></html>")   
    with open(report_file,'w') as ofil:
        ofil.write('\n'.join(html))

DIV_STYLE = """style = "display: block;border-style: solid; border-width: 5px;border-color:blue;padding:5px;margin:5px;" """

def cluster_div(prev,ids,lbl_lists):
    div = []    
    lbls = [lbl[0] for lbl in lbl_lists]
    lbls = list(np.unique(lbls))
    lbls.sort()
    for lbl in lbls:
        div.append(f'<div {DIV_STYLE}>\n<h1>Cluster{prev}{lbl}</h1>')        
        indexes = [ix for ix in range(len(ids)) if lbl_lists[ix][0]==lbl]
        current_indexes = [ix for ix in indexes if len(lbl_lists[ix]) == 1]
        next_indexes = [ix for ix in indexes if len(lbl_lists[ix]) > 1]
        for ix in current_indexes:
                div.append(f'<img src="images/{int(ids[ix])}.png" />')
        if len(next_indexes)>0:            
            #print(f'**{prev}**\n',indexes,'\n  ',current_indexes,'\n   ',next_indexes, len(next_indexes))        
            next_ids = [ids[ix] for ix in next_indexes]
            next_lbl_lists = [lbl_lists[ix][1:] for ix in next_indexes]
            #print('****',next_lbl_lists)
            div.append(cluster_div(f'{prev}{lbl}-',next_ids,next_lbl_lists))
        div.append('</div>')
    return '\n'.join(div)
    

def report_clusters_hierarchical(ixs,label_lists,report_file):
    html = ["""<!DOCTYPE html>
    <html lang="en">
       <head>
        <meta http-equiv="content-type" content="text/html; charset=utf-8">
        <meta charset="UTF-8">
        <title>Cluster Report</title>
       </head>
       <body>
       """]   
    html.append(cluster_div('',ixs,label_lists))   
    html.append("</body></html>")   
    with open(report_file,'w') as ofil:
        ofil.write('\n'.join(html))