import './styles.css';
export const metadata = { title: 'Atlas | Hybrid RAG', description: 'Grounded, searchable document intelligence' };
export default function Layout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
