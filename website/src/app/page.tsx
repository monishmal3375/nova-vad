import { Nav } from "@/components/Nav";
import { Hero } from "@/components/Hero";
import { Bento } from "@/components/Bento";
import { Benchmarks } from "@/components/Benchmarks";
import { Methodology } from "@/components/Methodology";
import { HowItWorks } from "@/components/HowItWorks";
import { Explainability } from "@/components/Explainability";
import { LiveDemo } from "@/components/LiveDemo";
import { QuickStart } from "@/components/QuickStart";
import { ContactCTA } from "@/components/ContactCTA";
import { About } from "@/components/About";
import { Footer } from "@/components/Footer";

export default function Home() {
  return (
    <>
      <Nav />
      <main className="flex-1">
        <Hero />
        <Bento />
        <Benchmarks />
        <Methodology />
        <HowItWorks />
        <Explainability />
        <LiveDemo />
        <QuickStart />
        <ContactCTA />
        <About />
      </main>
      <Footer />
    </>
  );
}
