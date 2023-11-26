import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
from tkinter import ttk
from matplotlib import colors as mcolors
from PIL import ImageTk, Image


class Kmeans(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.columnconfigure((0, 1), weight=1, minsize=640)
        self.rowconfigure(0, weight=1)

        self.config(bg="#1e1e1e")
        self.uiAttrib = []
        self.getAttribute("Wine.csv")
        self.buildUI()
        self.place(x=0, y=0, relheight=1, relwidth=1)

    # Get the attributes from the CSV File
    def getAttribute(self, fileName):
        with open(fileName, "r") as file:
            attribute = file.readline().strip()
            attribute = attribute.replace("_", " ")
            self.uiAttrib = attribute.split(",")

    # Get the data points based on the matching attribute
    def getPoints(self, fileName):
        # If equal return none
        if self.boxAttribute1.get() == self.boxAttribute2.get():
            print("Attribute 1 and 2 cannot be the same")
            return None

        try:
            # Open the file
            with open(fileName, mode="r") as file:
                attribs = csv.reader(file)
                header = next(attribs)

                # Get the index of the attributes
                self.firstAttrib = header.index(
                    self.boxAttribute1.get().replace(" ", "_")
                )
                self.secondAttrib = header.index(
                    self.boxAttribute2.get().replace(" ", "_")
                )

                # Get the points
                points = [
                    (float(row[self.firstAttrib]), float(row[self.secondAttrib]))
                    for row in attribs
                ]

            return points
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    # Get the value of N
    def getN(self):
        nValue = self.clusters.get()
        nValue = int(nValue)
        return nValue

    # Ge the distance between two points
    def getDistance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    # Function that performs the kmeans clustering
    def runFunction(self):
        # Get the points and N value
        points = self.getPoints("Wine.csv")
        k = self.getN()

        # If points or N is none return
        if not points or k is None:
            print("Kmeans failed to run")
            return

        points = np.array(points)

        # Get the title of the attributes
        xTitle = self.boxAttribute1.get().replace("-", " ")
        yTitle = self.boxAttribute2.get().replace("_", " ")

        # Print the title of the attributes
        print(f"Running Kmeans on {xTitle} and {yTitle}")

        # Get the initial centroids
        initialCentroids = points[np.random.choice(points.shape[0], k, replace=False)]

        # Copy the initial centroids
        centroids = initialCentroids.copy()

        # Limit to 100 iteration
        for _ in range(100):
            # Create a list of empty clusters
            clusters = [[] for _ in range(len(centroids))]
            for point in points:
                # Get the distance between the point and the centroid
                distances = [
                    self.getDistance(point, centroid) for centroid in centroids
                ]
                # Get the index of the closest centroid
                closestIndex = np.argmin(distances)
                # Append the point to the closest centroid
                clusters[closestIndex].append(point)

            # Get the new centroids
            newCentroids = np.array(
                [
                    # Get the mean of the cluster
                    np.mean(cluster, axis=0) if len(cluster) > 0 else np.random.rand(2)
                    for cluster in clusters
                ]
            )
            # If the centroids are the same break
            if np.all(centroids == newCentroids):
                break

            # Set the new centroids
            centroids = newCentroids

        # Clear the table
        for item in self.centroidTable.get_children():
            self.centroidTable.delete(item)

        # Print the centroids and clusters
        outputData = []
        for i, cluster in enumerate(clusters):
            centroidInfo = f"Centroid: {i} ({centroids[i][0]}, {centroids[i][1]}) [{initialCentroids[i][0]}, {initialCentroids[i][1]}]"
            self.centroidTable.insert("", "end", values=(centroidInfo,))
            outputData.append(centroidInfo)
            for point in cluster:
                outputData.append(str(list(point)))
                self.centroidTable.insert("", "end", values=(str(list(point)),))

        # Write the output to a file
        with open("output.csv", "w") as file:
            for line in outputData:
                file.write(line + "\n")

        # Create the scatter plot
        plt.figure(figsize=(8, 6))
        colors = list(mcolors.CSS4_COLORS.values())

        randomColors = random.sample(colors, len(clusters))

        for idx, cluster in enumerate(clusters):
            cluster = np.array(cluster)
            plt.scatter(
                cluster[:, 0],
                cluster[:, 1],
                color=randomColors[idx],
                label=f"Cluster {idx+1}",
            )

        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            color="black",
            marker="X",
            s=100,
            label="Centroids",
        )
        plt.title(f"KMeans Clustering on {xTitle} and {yTitle}")
        plt.xlabel(xTitle)
        plt.ylabel(yTitle)
        plt.legend()

        # Save the plot after it has been displayed
        plt.savefig("kmeans.png")
        plt.close()

        # Display the plot
        img = ImageTk.PhotoImage(Image.open("kmeans.png").resize((640, 640)))
        scatterImage = tk.Label(self.rightCol, image=img)
        scatterImage.image = img
        scatterImage.grid(row=1, column=0, padx=20, pady=20)

    # Reset the UI
    def resetFunction(self):
        self.buildUI()

    # Build the UI
    def buildUI(self):
        self.leftCol = tk.Frame(self, bg="#1e1e1e", bd=1, relief="solid")
        self.leftCol.columnconfigure(0, weight=1)
        self.leftCol.rowconfigure(0, weight=0)
        self.leftCol.rowconfigure(1, weight=1)

        self.rightCol = tk.Frame(self, bg="#1e1e1e")
        self.rightCol.columnconfigure(0, weight=1)
        self.rightCol.rowconfigure(0, weight=0)
        self.rightCol.rowconfigure(1, weight=1)

        self.upperContainer = tk.Frame(self.leftCol, bg="#1e1e1e", padx=20, pady=20)
        self.lowerContainer = tk.Frame(self.leftCol, bg="#1e1e1e")
        self.buttonContainer = tk.Frame(self.upperContainer, bg="#1e1e1e", pady=15)

        self.upperContainer.columnconfigure((0, 1), weight=1)
        self.upperContainer.rowconfigure((0, 1, 2, 3), weight=1)
        self.lowerContainer.columnconfigure((0, 1), weight=1)
        self.lowerContainer.rowconfigure((0), weight=1)
        self.buttonContainer.columnconfigure((0, 1), weight=1)
        self.buttonContainer.rowconfigure(0, weight=1)

        self.boxAttribute1 = ttk.Combobox(
            self.upperContainer,
            state="readonly",
            values=self.uiAttrib,
        )

        verticalScroll = ttk.Scrollbar(self.lowerContainer, orient="vertical")
        horizontalScroll = ttk.Scrollbar(self.lowerContainer, orient="horizontal")

        self.centroidTable = ttk.Treeview(
            self.lowerContainer,
            columns=("Centroid"),
            show="headings",
            yscrollcommand=verticalScroll.set,
            xscrollcommand=horizontalScroll.set,
        )

        verticalScroll.configure(command=self.centroidTable.yview)
        horizontalScroll.configure(command=self.centroidTable.xview)

        self.centroidTable.heading("Centroid", text="Centroid")
        self.centroidTable.column("Centroid", stretch=tk.YES)

        self.boxAttribute2 = ttk.Combobox(
            self.upperContainer,
            state="readonly",
            values=self.uiAttrib,
        )

        self.clusters = ttk.Combobox(
            self.upperContainer,
            state="readonly",
            values=[i + 1 for i in range(0, 10)],
        )

        self.selectAttribute1 = tk.Label(
            self.upperContainer,
            text=f"Select Attributes 1",
            font=("Roboto Mono Bold", 12),
            fg="white",
            bg="#1e1e1e",
            padx=10,
        )

        self.selectAttribute2 = tk.Label(
            self.upperContainer,
            text=f"Select Attributes 2",
            font=("Roboto Mono Bold", 12),
            fg="white",
            bg="#1e1e1e",
            padx=10,
        )

        self.enterCluster = tk.Label(
            self.upperContainer,
            text=f"Enter N Clusters",
            font=("Roboto Mono Bold", 12),
            fg="white",
            bg="#1e1e1e",
        )

        self.centroidLabel = tk.Label(
            self.lowerContainer,
            text=f"Centroid and Clusters",
            font=("Roboto Mono Bold", 12),
            justify="center",
            fg="white",
            bg="#1e1e1e",
        )

        self.scatterLabel = tk.Label(
            self.rightCol,
            text=f"Kmeans Scatter Plot",
            font=("Roboto Mono Bold", 12),
            justify="center",
            fg="white",
            bg="#1e1e1e",
        )

        self.runButton = tk.Button(
            self.buttonContainer,
            text="Run",
            font=("Roboto Mono", 12),
            bg="#007acc",
            fg="#1e1e1e",
            bd=0,
            command=self.runFunction,
        )

        self.resetButton = tk.Button(
            self.buttonContainer,
            text="Reset",
            font=("Roboto Mono", 12),
            bg="#007acc",
            fg="#1e1e1e",
            bd=0,
            command=self.resetFunction,
        )

        self.boxAttribute1.grid(row=0, column=1, sticky="nsew")
        self.boxAttribute2.grid(row=1, column=1, sticky="nsew")
        self.clusters.grid(row=2, column=1, sticky="nsew")
        self.selectAttribute1.grid(row=0, column=0, sticky="ns")
        self.selectAttribute2.grid(row=1, column=0, sticky="ns")
        self.centroidLabel.grid(row=0, column=0, sticky="nsew", padx=5)
        self.scatterLabel.grid(row=0, column=0, sticky="new", padx=20, pady=20)
        self.enterCluster.grid(row=2, column=0, sticky="ns")

        self.centroidTable.grid(row=0, column=1, sticky="nsew", pady=20, padx=20)

        self.runButton.grid(row=0, column=0, sticky="ew", padx=5)
        self.resetButton.grid(row=0, column=1, sticky="ew", padx=5)

        self.leftCol.grid(row=0, column=0, sticky="nsew")
        self.rightCol.grid(row=0, column=1, sticky="nsew")
        self.upperContainer.grid(row=0, column=0, sticky="new")
        self.lowerContainer.grid(row=1, column=0, sticky="nsew")
        self.buttonContainer.grid(row=3, column=0, sticky="ew")
