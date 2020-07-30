module FileParsing where

import System.IO
import Data.List.Split
import Safe
import Data.Maybe

type ImageData = [Bool] -- First we let the features simply be whether or not a given pixel contains anything: 1 for yes, 0 for no.
data LabeledImage = LabeledImage { rawData :: ImageData, label :: Int } deriving Show -- A labeled image is a pair of its raw data, represented as an array of booleans, and its label, represented by an integer, 0-9 for digits, and 0/1 for faces.
type DataFile = [LabeledImage] -- A pair of files, one with the image data, and one with the labels, will reduce to an array of labeled images.

digitRowLength = 28 -- Digits are 28x28.
faceRowLength = 70 -- Faces are 60x70.

readRawImageData :: String -> Int -> IO [[String]]
readRawImageData path size = do
        fileContents <- readFile path
        return $ chunksOf size $ lines fileContents

readRawDigitData :: String -> IO [[String]]
readRawDigitData = flip readRawImageData $ digitRowLength

readRawFaceData :: String -> IO [[String]]
readRawFaceData = flip readRawImageData $ faceRowLength

readLabelData :: String -> IO [Int]
readLabelData path = do
        fileContents <- readFile path
        return $ map (read :: String -> Int) (lines fileContents)

readLabeledImageData :: String -> String -> Int -> IO DataFile
readLabeledImageData imgPath labelPath size = do
        rawData <- readRawImageData imgPath size
        labels <- readLabelData labelPath
        return $ zipWith (\raw label -> LabeledImage (raw >>= map (/= ' ')) label) rawData labels

readLabeledDigitData :: IO DataFile
readLabeledDigitData = readLabeledImageData "data/digitdata/testimages" "data/digitdata/testlabels" digitRowLength

readLabeledFaceData :: IO DataFile
readLabeledFaceData = readLabeledImageData "data/facedata/facedatatest" "data/facedata/facedatatestlabels" faceRowLength

printImage :: [[String]] -> Int -> IO ()
printImage dat i = let img = atMay dat i in maybe (putStrLn "Out of bounds") (mapM_ putStrLn) img

readAndPrintImage :: String -> Int -> Int -> IO ()
readAndPrintImage path size i = readRawImageData path size >>= (flip printImage) i

readAndPrintDigit :: Int -> IO ()
readAndPrintDigit i = readRawDigitData "data/digitdata/testimages" >>= (flip printImage) i
