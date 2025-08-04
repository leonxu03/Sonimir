"use client";

import Link from "next/link";
import { useState } from "react";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Progress } from "~/components/ui/progress";
import { ColorScale } from "~/components/ui/ColorScale";
import FeatureMap from "~/components/ui/FeatureMap";
import Waveform from "~/components/ui/Waveform";

interface Prediction {
  class: string;
  confidence: number;
}

interface LayerData {
  shape: number[];
  values: number[][];
}

interface VisualizationData {
  [layerName: string]: LayerData; // eslint-disable-line
}

interface WaveformData {
  values: number[];
  sample_rate: number;
  duration: number;
}

interface ApiResponse {
  predictions: Prediction[];
  visualization: VisualizationData;
  input_spectrogram: LayerData;
  waveform: WaveformData;
}

const ESC50_EMOJI_MAP: Record<string, string> = {
  dog: "🐕",
  rain: "🌧️",
  crying_baby: "👶",
  door_wood_knock: "🚪",
  helicopter: "🚁",
  rooster: "🐓",
  sea_waves: "🌊",
  sneezing: "🤧",
  mouse_click: "🖱️",
  chainsaw: "🪚",
  pig: "🐷",
  crackling_fire: "🔥",
  clapping: "👏",
  keyboard_typing: "⌨️",
  siren: "🚨",
  cow: "🐄",
  crickets: "🦗",
  breathing: "💨",
  door_wood_creaks: "🚪",
  car_horn: "📯",
  frog: "🐸",
  chirping_birds: "🐦",
  coughing: "😷",
  can_opening: "🥫",
  engine: "🚗",
  cat: "🐱",
  water_drops: "💧",
  footsteps: "👣",
  washing_machine: "🧺",
  train: "🚂",
  hen: "🐔",
  wind: "💨",
  laughing: "😂",
  vacuum_cleaner: "🧹",
  church_bells: "🔔",
  insects: "🦟",
  pouring_water: "🚰",
  brushing_teeth: "🪥",
  clock_alarm: "⏰",
  airplane: "✈️",
  sheep: "🐑",
  toilet_flush: "🚽",
  snoring: "😴",
  clock_tick: "⏱️",
  fireworks: "🎆",
  crow: "🐦‍⬛",
  thunderstorm: "⛈️",
  drinking_sipping: "🥤",
  glass_breaking: "🔨",
  hand_saw: "🪚",
};

const getEmojiForClass = (className: string): string => {
  return ESC50_EMOJI_MAP[className] ?? "📢";
};

function splitLayers(visualization: VisualizationData) {
  const main: [string, LayerData][] = [];
  const internals: Record<string, [string, LayerData][]> = {};

  for (const [name, data] of Object.entries(visualization)) {
    if (!name.includes(".")) {
      main.push([name, data]);
    } else {
      const [parent] = name.split(".");
      if (parent === undefined) {
        continue;
      }

      internals[parent] ??= [];

      internals[parent].push([name, data]);
    }
  }

  return { main, internals };
}

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [vizData, setVizData] = useState<ApiResponse | null>(null);

  console.log("HELLO");
  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setIsLoading(true);
    setError(null);
    setVizData(null);

    console.log("HERE 1");

    const reader = new FileReader();
    reader.readAsArrayBuffer(file);
    reader.onload = async () => {
      try {
        const arrayBuffer = reader.result as ArrayBuffer;
        const base64String = btoa(
          new Uint8Array(arrayBuffer).reduce(
            (data, byte) => data + String.fromCharCode(byte),
            "",
          ),
        );

        const response = await fetch(
          "https://leonxu03--sonimir-audioclassifier-inference.modal.run/",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ audio_data: base64String }),
          },
        );

        console.log("RESPONSE", response);

        if (!response.ok) {
          throw new Error(`API error ${response.statusText}`);
        }

        const data: ApiResponse = await response.json();
        setVizData(data);
      } catch (err: unknown) {
        setError(
          err instanceof Error ? err.message : "An unexpected error occurred",
        );
      } finally {
        setIsLoading(false);
      }
    };
    reader.onerror = () => {
      setError("Failed to read file");
      setIsLoading(false);
    };
  };

  const { main, internals } = vizData
    ? splitLayers(vizData?.visualization)
    : { main: [], internals: {} };

  return (
    <main
      className="min-h-screen animate-[pulseGradient_12s_ease-in-out_infinite] bg-gradient-to-r from-rose-200 via-sky-200 to-indigo-300 pb-8"
      style={{
        backgroundSize: "200% 200%",
        backgroundPosition: "0% 50%",
      }}
    >
      <header className="mb-16 flex h-16 items-center justify-between px-4">
        <div className="flex items-center gap-4">
          <button className="text-stone-700">
            <span className="sr-only">Toggle Menu</span>☰
          </button>
          <span className="font-medium text-stone-700">Menu</span>
        </div>
        <div className="flex items-center gap-4">
          <input
            type="text"
            placeholder="Search..."
            className="rounded-md bg-white px-3 py-1 text-sm"
          />
          <Link href="https://github.com/leonxu03/Sonimir">
            <img
              src="/github.png"
              className="h-6 w-6 opacity-80 transition-opacity hover:opacity-100"
            />
          </Link>
        </div>
      </header>
      <div className="mx-auto max-w-[100%]">
        <section className="py-12 text-center">
          <span className="mb-2 inline-block rounded-full bg-stone-100 px-3 py-1 text-xs text-stone-600">
            Audio Classifier Demo
          </span>
          <h1 className="ext-4xl mt-2 font-bold tracking-tight text-stone-900 sm:text-5xl">
            Sonimir
          </h1>
          <p className="mt-4 text-lg text-stone-700">
            Audio Visualizer + Convolutional Neural Network
          </p>
          <p className="text-md mt-2 text-stone-600">
            {
              "Upload a WAV file to see the model's predictions and feature maps"
            }
          </p>
          <div className="mt-6 flex justify-center gap-4">
            <div className="flex flex-col items-center">
              <label className="relative inline-block">
                <input
                  onChange={handleFileChange}
                  disabled={isLoading}
                  type="file"
                  accept=".wav"
                  className="absolute inset-0 h-full w-full opacity-0"
                />
                <Button disabled={isLoading}>
                  {isLoading ? "Analyzing..." : "Get Started"}
                </Button>
              </label>
              {fileName && (
                <Badge
                  variant="secondary"
                  className="my-4 bg-stone-200 text-stone-700"
                >
                  {fileName}
                </Badge>
              )}
            </div>
          </div>
        </section>

        {error && (
          <div className="fixed right-8 bottom-8 z-50">
            <Card className="w-auto max-w-xs border-red-200 bg-red-50 p-2 shadow-lg">
              <CardContent>
                <p className="text-red-600">{`Error: ${error}`}</p>
              </CardContent>
            </Card>
          </div>
        )}

        {vizData && (
          <div className="mx-24 space-y-8">
            <Card>
              <CardHeader>
                <CardTitle className="text-stone-900">
                  Top Predictions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-8">
                  {vizData.predictions.slice(0, 3).map((pred, i) => (
                    <div key={pred.class} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="text-md font-medium text-stone-700">
                          {getEmojiForClass(pred.class)}{" "}
                          <span>{pred.class.replaceAll("_", " ")}</span>
                        </div>
                        <Badge variant={i === 0 ? "default" : "secondary"}>
                          {(pred.confidence * 100).toFixed(1)}%
                        </Badge>
                      </div>
                      <Progress
                        value={pred.confidence * 100}
                        className="h-2"
                      ></Progress>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Card>
                <CardHeader className="text-stone-900">
                  <CardTitle className="text-stone-900">
                    Input Spectrogram
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <FeatureMap
                    data={vizData.input_spectrogram.values}
                    title={`${vizData.input_spectrogram.shape.join(" x ")}`}
                    spectrogram
                  ></FeatureMap>
                  <div className="mt-5 flex justify-end">
                    <ColorScale width={200} height={16} min={-1} max={1} />
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-stone-900">
                    Audio Waveform
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Waveform
                    data={vizData.waveform.values}
                    title={`${vizData.waveform.duration.toFixed(2)}s at ${vizData.waveform.sample_rate}Hz`}
                  />
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Convolutional Layer Outputs</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-5 gap-6">
                  {main.map(([mainName, mainData]) => (
                    <div key={mainName} className="space-y-4">
                      <div>
                        <h4 className="mb-2 font-medium text-stone-700">
                          {mainName}
                        </h4>
                        <FeatureMap
                          data={mainData.values}
                          title={`${mainData.shape.join(" x ")}`}
                        ></FeatureMap>
                      </div>
                      {internals[mainName] && (
                        <div className="h-80 overflow-y-auto rounded border border-stone-200 bg-stone-50 p-2">
                          <div className="space-y-2">
                            {internals[mainName]
                              .sort(([a], [b]) => a.localeCompare(b))
                              .map(([layerName, layerData]) => (
                                <FeatureMap
                                  key={layerName}
                                  data={layerData.values}
                                  title={layerName.replace(`${mainName}.`, "")}
                                  internal={true}
                                />
                              ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
                <div className="mt-5 flex justify-end">
                  <ColorScale width={200} height={16} min={-1} max={1} />
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </main>
  );
}
